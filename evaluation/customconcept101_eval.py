"""
CustomConcept101 Text-to-Image Inference & Evaluation for MedMax.

Usage:
  # Single-concept inference + CLIP scoring
  CUDA_VISIBLE_DEVICES=0 python -m evaluation.customconcept101_eval \
      --ckpt <chameleon_ckpt_dir> \
      --dataset_root /data1/data/kangborui/zhongyukun/medmax/customconcept101 \
      --save_dir evaluation/outputs/customconcept101 \
      --mode single

  # Multi-concept inference + CLIP scoring
  CUDA_VISIBLE_DEVICES=0 python -m evaluation.customconcept101_eval \
      --ckpt <chameleon_ckpt_dir> \
      --dataset_root /data1/data/kangborui/zhongyukun/medmax/customconcept101 \
      --save_dir evaluation/outputs/customconcept101 \
      --mode multi

  # Both single + multi
  CUDA_VISIBLE_DEVICES=0 python -m evaluation.customconcept101_eval \
      --ckpt <chameleon_ckpt_dir> \
      --dataset_root /data1/data/kangborui/zhongyukun/medmax/customconcept101 \
      --save_dir evaluation/outputs/customconcept101 \
      --mode both

  # Use --prompt_processor sft  for SFT-mode prompting
  # Use --max_concepts N        to limit how many concepts to run (for debugging)
  # Use --clip_model <name>     to change CLIP model (default: openai ViT-L/14)
"""

import argparse
import json
import os
import glob
import numpy as np
import torch
import random
import transformers
from tqdm import tqdm
from PIL import Image

from inference.inference_utils import load_chameleon, chameleon_generate
from evaluation.eval_utils import (
    set_seeds,
    add_results_to_json,
    log_samples,
    calculate_accuracy_and_stderr,
    CLIPSimilarity,
    SimilarityType,
)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_single_concept_data(dataset_root: str):
    """Load dataset.json and resolve paths / prompts."""
    dataset_json = os.path.join(dataset_root, "dataset.json")
    with open(dataset_json, "r") as f:
        entries = json.load(f)

    concepts = []
    for entry in entries:
        # Resolve prompt file
        prompt_path = os.path.join(dataset_root, entry["prompt_filename"])
        with open(prompt_path, "r") as pf:
            prompt_templates = [line.strip() for line in pf if line.strip()]

        # Resolve reference images
        img_dir = os.path.join(dataset_root, entry["instance_data_dir"])
        ref_images = sorted(glob.glob(os.path.join(img_dir, "*")))
        ref_images = [p for p in ref_images if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))]

        # Build concept name by replacing <new1> placeholder
        instance_prompt = entry["instance_prompt"]  # e.g. "photo of a <new1> chair"
        class_prompt = entry["class_prompt"]         # e.g. "chair"

        # Derive a readable concept name from instance_data_dir
        concept_id = os.path.basename(entry["instance_data_dir"])  # e.g. "furniture_chair1"

        concepts.append({
            "concept_id": concept_id,
            "instance_prompt": instance_prompt,
            "class_prompt": class_prompt,
            "prompt_templates": prompt_templates,
            "ref_images": ref_images,
            "img_dir": img_dir,
        })

    return concepts


def load_multi_concept_data(dataset_root: str):
    """Load dataset_multiconcept.json and resolve paths / prompts."""
    dataset_json = os.path.join(dataset_root, "dataset_multiconcept.json")
    with open(dataset_json, "r") as f:
        groups = json.load(f)

    multi_concepts = []
    for group in groups:
        concept1 = group[0]
        concept2 = group[1]
        compose_info = group[2]

        compose_prompt_path = os.path.join(dataset_root, compose_info["prompt_filename_compose"])
        with open(compose_prompt_path, "r") as pf:
            compose_templates = [line.strip() for line in pf if line.strip()]

        # Also load individual prompts
        def _load_concept(entry):
            prompt_path = os.path.join(dataset_root, entry["prompt_filename"])
            with open(prompt_path, "r") as pf:
                templates = [line.strip() for line in pf if line.strip()]
            img_dir = os.path.join(dataset_root, entry["instance_data_dir"])
            ref_images = sorted(glob.glob(os.path.join(img_dir, "*")))
            ref_images = [p for p in ref_images if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))]
            return {
                "concept_id": os.path.basename(entry["instance_data_dir"]),
                "instance_prompt": entry["instance_prompt"],
                "class_prompt": entry["class_prompt"],
                "prompt_templates": templates,
                "ref_images": ref_images,
                "img_dir": img_dir,
            }

        multi_concepts.append({
            "concept1": _load_concept(concept1),
            "concept2": _load_concept(concept2),
            "compose_templates": compose_templates,
        })

    return multi_concepts


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_single_prompts(concept, use_instance_prompt=False):
    """
    For each template like 'photo of a {}.', fill in the concept descriptor.
    By default we use the class_prompt (e.g. 'chair').
    If use_instance_prompt, we extract the descriptor from instance_prompt.
    """
    class_prompt = concept["class_prompt"]
    prompts = []
    for template in concept["prompt_templates"]:
        prompt = template.replace("{}", class_prompt)
        prompts.append(prompt)
    return prompts


def build_multi_prompts(multi_entry):
    """
    For compose templates like 'photo of the {0} and {1}.',
    fill {0} with concept1 class_prompt and {1} with concept2 class_prompt.
    """
    c1 = multi_entry["concept1"]["class_prompt"]
    c2 = multi_entry["concept2"]["class_prompt"]
    prompts = []
    for template in multi_entry["compose_templates"]:
        prompt = template.replace("{0}", c1).replace("{1}", c2)
        prompts.append(prompt)
    return prompts


# ---------------------------------------------------------------------------
# CLIP scoring helpers
# ---------------------------------------------------------------------------

def get_clip_model(model_name="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"):
    """Get a CLIPSimilarity model. Uses BiomedCLIP by default (same as MedMax)."""
    return CLIPSimilarity(model_name=model_name)


def compute_clip_text_image_score(clip_model, image_path, text):
    """Compute CLIP similarity between generated image and the prompt text."""
    return clip_model.calculate_similarity(image_path, text, SimilarityType.IMAGE_CAPTION)


def compute_clip_image_image_score(clip_model, gen_image_path, ref_image_paths):
    """Compute average CLIP image-image similarity between generated and reference images."""
    scores = []
    for ref_path in ref_image_paths:
        score = clip_model.calculate_similarity(gen_image_path, ref_path, SimilarityType.IMAGE_IMAGE)
        if score is not None:
            scores.append(score)
    return float(np.mean(scores)) if scores else None


# ---------------------------------------------------------------------------
# Single-concept inference + evaluation
# ---------------------------------------------------------------------------

def run_single_concept_eval(
    model,
    clip_model,
    concepts,
    save_dir,
    save_name,
    prompt_processor_name="chameleon",
    sft=False,
    max_gen_len=60,
    max_concepts=None,
    images_per_concept=4,
    start_from=None,
):
    """
    For each concept:
      1. Build prompts from templates
      2. Generate images via chameleon_generate (text-to-image)
      3. Compute CLIP text-image similarity (prompt faithfulness)
      4. Compute CLIP image-image similarity (concept fidelity vs ref images)
    """
    all_text_image_scores = []
    all_image_image_scores = []
    all_samples = []
    per_concept_results = {}

    if max_concepts is not None:
        concepts = concepts[:max_concepts]

    # ← 跳过 start_from 之前的概念
    if start_from is not None:
        concept_ids = [c["concept_id"] for c in concepts]
        if start_from in concept_ids:
            start_idx = concept_ids.index(start_from)
            print(f"Skipping first {start_idx} concepts, starting from '{start_from}'")
            concepts = concepts[start_idx:]
        else:
            print(f"[WARN] start_from='{start_from}' not found in concepts, running all.")

    for concept in tqdm(concepts, desc="Single-concept inference"):
        concept_id = concept["concept_id"]
        prompts = build_single_prompts(concept)
        prompts = prompts[:images_per_concept]
        ref_images = concept["ref_images"]

        concept_ti_scores = []
        concept_ii_scores = []
        concept_samples = []

        concept_save_dir = os.path.join(save_dir, "inference", "single", concept_id)

        for prompt_idx, prompt in enumerate(tqdm(prompts, desc=f"  {concept_id}", leave=False)):
            # Generate image
            content = [prompt]
            modality = ["text"]

            try:
                outputs = chameleon_generate(
                    model,
                    content=content,
                    modality=modality,
                    task="image-gen",
                    sft=sft,
                    max_gen_len=max_gen_len,
                    save_dir=concept_save_dir,
                )
                gen_image_path = outputs[0]
            except Exception as e:
                print(f"  [WARN] Generation failed for concept={concept_id}, prompt_idx={prompt_idx}: {e}")
                continue

            # CLIP text-image score (prompt faithfulness)
            ti_score = compute_clip_text_image_score(clip_model, gen_image_path, prompt)
            if ti_score is not None:
                concept_ti_scores.append(ti_score)
                all_text_image_scores.append(ti_score)

            # CLIP image-image score (concept fidelity)
            ii_score = compute_clip_image_image_score(clip_model, gen_image_path, ref_images)
            if ii_score is not None:
                concept_ii_scores.append(ii_score)
                all_image_image_scores.append(ii_score)

            sample = {
                "concept_id": concept_id,
                "prompt": prompt,
                "generated_image": gen_image_path,
                "clip_text_image": ti_score,
                "clip_image_image": ii_score,
            }
            concept_samples.append(sample)
            all_samples.append(sample)

        # Per-concept results
        if concept_ti_scores:
            avg_ti, std_ti = calculate_accuracy_and_stderr(concept_ti_scores)
        else:
            avg_ti, std_ti = 0.0, 0.0
        if concept_ii_scores:
            avg_ii, std_ii = calculate_accuracy_and_stderr(concept_ii_scores)
        else:
            avg_ii, std_ii = 0.0, 0.0

        per_concept_results[concept_id] = {
            "clip_text_image_mean": avg_ti,
            "clip_text_image_stderr": std_ti,
            "clip_image_image_mean": avg_ii,
            "clip_image_image_stderr": std_ii,
            "num_prompts": len(prompts),
            "num_generated": len(concept_ti_scores),
        }

    # Overall results
    if all_text_image_scores:
        overall_ti_mean, overall_ti_stderr = calculate_accuracy_and_stderr(all_text_image_scores)
    else:
        overall_ti_mean, overall_ti_stderr = 0.0, 0.0
    if all_image_image_scores:
        overall_ii_mean, overall_ii_stderr = calculate_accuracy_and_stderr(all_image_image_scores)
    else:
        overall_ii_mean, overall_ii_stderr = 0.0, 0.0

    results = {
        "customconcept101_single": {
            "overall": {
                "clip_text_image_mean": overall_ti_mean,
                "clip_text_image_stderr": overall_ti_stderr,
                "clip_image_image_mean": overall_ii_mean,
                "clip_image_image_stderr": overall_ii_stderr,
                "total_concepts": len(concepts),
                "total_images_generated": len(all_text_image_scores),
            },
            "per_concept": per_concept_results,
        }
    }

    # Save
    os.makedirs(os.path.join(save_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "logs"), exist_ok=True)

    save_path = os.path.join(save_dir, "results", f"{save_name}.json")
    add_results_to_json(save_path, results)

    log_samples(
        os.path.join(save_dir, "logs", save_name),
        "customconcept101_single",
        all_samples,
    )

    print(f"\n[Single-Concept Results]")
    print(f"  CLIP Text-Image:  {overall_ti_mean:.4f} ± {overall_ti_stderr:.4f}")
    print(f"  CLIP Image-Image: {overall_ii_mean:.4f} ± {overall_ii_stderr:.4f}")
    print(f"  Total generated:  {len(all_text_image_scores)}")

    return results


# ---------------------------------------------------------------------------
# Multi-concept inference + evaluation
# ---------------------------------------------------------------------------

def run_multi_concept_eval(
    model,
    clip_model,
    multi_concepts,
    save_dir,
    save_name,
    prompt_processor_name="chameleon",
    sft=False,
    max_gen_len=60,
    max_concepts=None,
    images_per_concept=4,
):
    """
    For each multi-concept group:
      1. Build composed prompts from templates (两个概念组合)
      2. Generate images
      3. Compute CLIP text-image similarity
      4. Compute CLIP image-image similarity vs both concepts' reference images
    """
    all_text_image_scores = []
    all_ii_concept1_scores = []
    all_ii_concept2_scores = []
    all_samples = []
    per_group_results = {}

    if max_concepts is not None:
        multi_concepts = multi_concepts[:max_concepts]

    for group_idx, group in enumerate(tqdm(multi_concepts, desc="Multi-concept inference")):
        c1 = group["concept1"]
        c2 = group["concept2"]
        group_id = f"{c1['concept_id']}__x__{c2['concept_id']}"

        compose_prompts = build_multi_prompts(group)
        compose_prompts = compose_prompts[:images_per_concept]

        group_ti_scores = []
        group_ii_c1_scores = []
        group_ii_c2_scores = []
        group_samples = []

        group_save_dir = os.path.join(save_dir, "inference", "multi", group_id)

        for prompt_idx, prompt in enumerate(tqdm(compose_prompts, desc=f"  {group_id}", leave=False)):
            content = [prompt]
            modality = ["text"]

            try:
                outputs = chameleon_generate(
                    model,
                    content=content,
                    modality=modality,
                    task="image-gen",
                    sft=sft,
                    max_gen_len=max_gen_len,
                    save_dir=group_save_dir,
                )
                gen_image_path = outputs[0]
            except Exception as e:
                print(f"  [WARN] Generation failed for group={group_id}, prompt_idx={prompt_idx}: {e}")
                continue

            # CLIP text-image
            ti_score = compute_clip_text_image_score(clip_model, gen_image_path, prompt)
            if ti_score is not None:
                group_ti_scores.append(ti_score)
                all_text_image_scores.append(ti_score)

            # CLIP image-image vs concept1 refs
            ii_c1 = compute_clip_image_image_score(clip_model, gen_image_path, c1["ref_images"])
            if ii_c1 is not None:
                group_ii_c1_scores.append(ii_c1)
                all_ii_concept1_scores.append(ii_c1)

            # CLIP image-image vs concept2 refs
            ii_c2 = compute_clip_image_image_score(clip_model, gen_image_path, c2["ref_images"])
            if ii_c2 is not None:
                group_ii_c2_scores.append(ii_c2)
                all_ii_concept2_scores.append(ii_c2)

            sample = {
                "group_id": group_id,
                "concept1": c1["concept_id"],
                "concept2": c2["concept_id"],
                "prompt": prompt,
                "generated_image": gen_image_path,
                "clip_text_image": ti_score,
                "clip_image_image_c1": ii_c1,
                "clip_image_image_c2": ii_c2,
            }
            group_samples.append(sample)
            all_samples.append(sample)

        # Per-group results
        def _safe_stats(scores):
            if scores:
                return calculate_accuracy_and_stderr(scores)
            return 0.0, 0.0

        avg_ti, std_ti = _safe_stats(group_ti_scores)
        avg_ii_c1, std_ii_c1 = _safe_stats(group_ii_c1_scores)
        avg_ii_c2, std_ii_c2 = _safe_stats(group_ii_c2_scores)

        per_group_results[group_id] = {
            "clip_text_image_mean": avg_ti,
            "clip_text_image_stderr": std_ti,
            "clip_image_image_c1_mean": avg_ii_c1,
            "clip_image_image_c1_stderr": std_ii_c1,
            "clip_image_image_c2_mean": avg_ii_c2,
            "clip_image_image_c2_stderr": std_ii_c2,
            "num_prompts": len(compose_prompts),
            "num_generated": len(group_ti_scores),
        }

    # Overall
    def _safe_stats(scores):
        if scores:
            return calculate_accuracy_and_stderr(scores)
        return 0.0, 0.0

    overall_ti_mean, overall_ti_stderr = _safe_stats(all_text_image_scores)
    overall_ii_c1_mean, overall_ii_c1_stderr = _safe_stats(all_ii_concept1_scores)
    overall_ii_c2_mean, overall_ii_c2_stderr = _safe_stats(all_ii_concept2_scores)

    # Combined image-image score (average across both concepts)
    all_ii_combined = all_ii_concept1_scores + all_ii_concept2_scores
    overall_ii_combined_mean, overall_ii_combined_stderr = _safe_stats(all_ii_combined)

    results = {
        "customconcept101_multi": {
            "overall": {
                "clip_text_image_mean": overall_ti_mean,
                "clip_text_image_stderr": overall_ti_stderr,
                "clip_image_image_c1_mean": overall_ii_c1_mean,
                "clip_image_image_c1_stderr": overall_ii_c1_stderr,
                "clip_image_image_c2_mean": overall_ii_c2_mean,
                "clip_image_image_c2_stderr": overall_ii_c2_stderr,
                "clip_image_image_combined_mean": overall_ii_combined_mean,
                "clip_image_image_combined_stderr": overall_ii_combined_stderr,
                "total_groups": len(multi_concepts),
                "total_images_generated": len(all_text_image_scores),
            },
            "per_group": per_group_results,
        }
    }

    # Save
    os.makedirs(os.path.join(save_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "logs"), exist_ok=True)

    save_path = os.path.join(save_dir, "results", f"{save_name}.json")
    add_results_to_json(save_path, results)

    log_samples(
        os.path.join(save_dir, "logs", save_name),
        "customconcept101_multi",
        all_samples,
    )

    print(f"\n[Multi-Concept Results]")
    print(f"  CLIP Text-Image:        {overall_ti_mean:.4f} ± {overall_ti_stderr:.4f}")
    print(f"  CLIP Image-Image (C1):  {overall_ii_c1_mean:.4f} ± {overall_ii_c1_stderr:.4f}")
    print(f"  CLIP Image-Image (C2):  {overall_ii_c2_mean:.4f} ± {overall_ii_c2_stderr:.4f}")
    print(f"  CLIP Image-Image (Avg): {overall_ii_combined_mean:.4f} ± {overall_ii_combined_stderr:.4f}")
    print(f"  Total generated:        {len(all_text_image_scores)}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(description="CustomConcept101 Text-to-Image Inference & Evaluation on MedMax")
    parser.add_argument("--ckpt", required=True, type=str,
                        help="Path to chameleon-format checkpoint directory")
    parser.add_argument("--dataset_root", required=True, type=str,
                        help="Path to customconcept101 dataset root "
                             "(containing dataset.json, prompts/, benchmark_dataset/)")
    parser.add_argument("--save_dir", default="evaluation/outputs/customconcept101", type=str,
                        help="Directory to save generated images, logs, and results")
    parser.add_argument("--save_name", default="customconcept101_results", type=str,
                        help="Name for the results/log files")
    parser.add_argument("--mode", default="single", choices=["single", "multi", "both"],
                        help="Run single-concept, multi-concept, or both")
    parser.add_argument("--prompt_processor", default="chameleon", choices=["chameleon", "sft"],
                        help="Prompt processor style (chameleon=no sft, sft=with sft tokens)")
    parser.add_argument("--clip_model", default="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
                        type=str, help="CLIP model for scoring")
    parser.add_argument("--max_gen_len", default=60, type=int,
                        help="Max generation length for image tokens")
    parser.add_argument("--max_concepts", default=None, type=int,
                        help="Limit number of concepts to process (for debugging)")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--images_per_concept", default=4, type=int,
                        help="Number of images to generate per concept (default: 4)")
    parser.add_argument("--start_from", default=None, type=str,
                        help="concept_id to start from (skip all concepts before it), e.g. 'plushie_happysad'")
    return parser.parse_args()


def main():
    args = parse_arguments()

    print("=" * 60)
    print("CustomConcept101 Inference & Evaluation")
    print("=" * 60)
    print(f"  Checkpoint:    {args.ckpt}")
    print(f"  Dataset root:  {args.dataset_root}")
    print(f"  Save dir:      {args.save_dir}")
    print(f"  Mode:          {args.mode}")
    print(f"  Prompt style:  {args.prompt_processor}")
    print(f"  CLIP model:    {args.clip_model}")
    print(f"  Max gen len:   {args.max_gen_len}")
    print(f"  Max concepts:  {args.max_concepts}")
    print("=" * 60)

    # Set seeds
    set_seeds(args.seed)

    # Load model
    print("\nLoading Chameleon model...")
    model = load_chameleon(args.ckpt)

    # Load CLIP model for scoring
    print("Loading CLIP model for evaluation...")
    clip_model = CLIPSimilarity(model_name=args.clip_model)

    # Determine sft mode
    sft = (args.prompt_processor == "sft")

    # Run single-concept evaluation
    if args.mode in ("single", "both"):
        print("\n" + "=" * 60)
        print("Running Single-Concept Evaluation")
        print("=" * 60)
        concepts = load_single_concept_data(args.dataset_root)
        print(f"Loaded {len(concepts)} single concepts")
        run_single_concept_eval(
            model=model,
            clip_model=clip_model,
            concepts=concepts,
            save_dir=args.save_dir,
            save_name=args.save_name,
            prompt_processor_name=args.prompt_processor,
            sft=sft,
            max_gen_len=args.max_gen_len,
            max_concepts=args.max_concepts,
            images_per_concept=args.images_per_concept,
            start_from=args.start_from,   # ← 传入
        )

    # Run multi-concept evaluation
    if args.mode in ("multi", "both"):
        print("\n" + "=" * 60)
        print("Running Multi-Concept Evaluation")
        print("=" * 60)
        multi_concepts = load_multi_concept_data(args.dataset_root)
        print(f"Loaded {len(multi_concepts)} multi-concept groups")
        run_multi_concept_eval(
            model=model,
            clip_model=clip_model,
            multi_concepts=multi_concepts,
            save_dir=args.save_dir,
            save_name=args.save_name,
            prompt_processor_name=args.prompt_processor,
            sft=sft,
            max_gen_len=args.max_gen_len,
            max_concepts=args.max_concepts,
            images_per_concept=args.images_per_concept,
        )

    print("\nDone! Results saved to:", os.path.join(args.save_dir, "results", f"{args.save_name}.json"))


if __name__ == "__main__":
    main()
