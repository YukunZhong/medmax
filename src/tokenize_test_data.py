#!/usr/bin/env python3
"""
Tokenize test-set images for ScienceQA / TextVQA / GQA using Anole's VQGAN,
producing JSONL files with the same schema as the training data:

    {
        "question_id": ...,
        "image_path": "...",
        "image_tokens": [int x 1024],   # VQGAN codebook ids (0-8191)
        "text": "...",
        "answer": "...",
        "tokens": [int ...]             # final merged BPE sequence
    }

Pipeline (identical to training data preparation):
    1. Load test.json from MoDE-official/instructions/{Task}/test.json
       (fall back to MoDE-official/data/{Task}/*_data.jsonl if needed)
    2. For every sample that contains an image, load the raw image from
       --image_base_dir (e.g. coin_raw/) and encode it with Anole's VQGAN
       to obtain 1024 codebook indices.
    3. Tokenize the text with Anole's text tokenizer, replace the <image>
       placeholder with [BOI] + offset_image_tokens + [EOI], and produce
       the final token sequence.
    4. Write each sample as a line in the output JSONL file.

Usage:
    python -m src.tokenize_test_data \
        --anole_dir   /path/to/Anole-7b-v0.1 \
        --data_dir    /path/to/MoDE-official \
        --image_base_dir /path/to/coin_raw \
        --output_dir  /path/to/MoDE-official/data \
        --tasks scienceqa textvqa gqa \
        --batch_size 16 \
        --device cuda

    # Or with torchrun for multi-GPU:
    torchrun --nproc-per-node=2 --standalone -m src.tokenize_test_data \
        --anole_dir ... --distributed
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import PIL
import PIL.Image
import torch
from tokenizers import Tokenizer
from tqdm import tqdm

# ── Make chameleon importable ─────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent.parent  # medmax/
sys.path.insert(0, str(SCRIPT_DIR))

from chameleon.inference.image_tokenizer import ImageTokenizer

# ──────────────────────────────────────────────────────────────────────
# Constants – must match medmax/src/tokenization.py
# ──────────────────────────────────────────────────────────────────────
IMAGE_TOKEN_OFFSET = 4          # VQGAN codebook 0-8191 → BPE 4-8195
BOI_TOKEN = 8197                # <racm3:break>  begin-of-image
EOI_TOKEN = 8196                # <eoss>         end-of-image
BOS_TOKEN = 0                   # <s>
EOS_TOKEN = 2                   # </s>
PAD_TOKEN = 1                   # <pad>
# Task configs ─────────────────────────────────────────────────────────
TASK_CONFIGS = {
    "scienceqa": {
        "instruction_test": "instructions/ScienceQA/test.json",
        "data_test":        "data/ScienceQA/test_data.jsonl",
        "output_name":      "ScienceQA",
        "image_subdir":     "",          # images are referenced as ScienceQA/images/...
    },
    "textvqa": {
        "instruction_test": "instructions/TextVQA/test.json",
        "data_fallback":    "data/TextVQA/val_data.jsonl",
        "output_name":      "TextVQA",
        "image_subdir":     "",
    },
    "gqa": {
        "instruction_test": "instructions/GQA/test.json",
        "data_test":        "data/GQA/test_data.jsonl",
        "output_name":      "GQA",
        "image_subdir":     "",
    },
    "imagenet": {
        "instruction_test": "instructions/ImageNet/test.json",
        "data_test":        "data/ImageNet/test_data.jsonl",
        "output_name":      "ImageNet",
        "image_subdir":     "",
        # Actual images are flat in coin_raw/ImageNet/<filename>.JPEG
        # The instruction file records paths as ImageNet_withlabel/val/<filename>.JPEG
        "image_dir_override": "ImageNet",
    },
}


# ──────────────────────────────────────────────────────────────────────
# Image preprocessing (same as VQVAEImageProcessor / ImageTokenizer)
# ──────────────────────────────────────────────────────────────────────
def whiten_transparency(img: PIL.Image.Image) -> PIL.Image.Image:
    if img.mode == "RGB":
        return img
    vals_rgba = np.array(img.convert("RGBA"))
    if not (vals_rgba[:, :, 3] < 255).any():
        return img.convert("RGB")
    alpha = vals_rgba[:, :, 3] / 255.0
    vals_rgb = (1 - alpha[:, :, np.newaxis]) * 255 + alpha[:, :, np.newaxis] * vals_rgba[:, :, :3]
    return PIL.Image.fromarray(vals_rgb.astype("uint8"), "RGB")


def preprocess_image(img: PIL.Image.Image, target_size: int = 512) -> torch.Tensor:
    """Resize + center-crop + normalise to [-1, 1], return (C, H, W) tensor."""
    img = whiten_transparency(img)
    s = min(img.size)
    scale = target_size / s
    new_size = (round(scale * img.size[0]), round(scale * img.size[1]))
    img = img.resize(new_size, PIL.Image.LANCZOS)
    x0 = (img.width - target_size) // 2
    y0 = (img.height - target_size) // 2
    img = img.crop((x0, y0, x0 + target_size, y0 + target_size))
    np_img = np.array(img) / 255.0
    np_img = np_img * 2 - 1
    return torch.from_numpy(np_img).permute(2, 0, 1).float()


# ──────────────────────────────────────────────────────────────────────
# Token helpers (match tokenization.py exactly)
# ──────────────────────────────────────────────────────────────────────
def offset_image_tokens(tokens: List[int]) -> List[int]:
    """Shift VQGAN codebook ids (0-8191) → BPE range (4-8195)."""
    return [t + IMAGE_TOKEN_OFFSET for t in tokens]


def build_full_token_sequence(
    text: str,
    image_tokens: Optional[List[int]],
    tokenizer: Tokenizer,
) -> List[int]:
    """
    Tokenize *text* with the Anole text tokenizer, and – if the text
    contains an <image> placeholder – splice in the image BPE tokens
    wrapped by BOI / EOI markers.

    The <image> placeholder is NOT a single token in Anole's vocabulary;
    it encodes to multiple sub-tokens.  We therefore split the raw text
    on the literal string "<image>", tokenize each fragment separately,
    and insert [BOI] + offset_image_tokens + [EOI] at the split point.

    Returns the complete token sequence: [BOS, ..., EOS].
    """
    if image_tokens is None or "<image>" not in text:
        return [BOS_TOKEN] + tokenizer.encode(text).ids + [EOS_TOKEN]

    # Prepare image segment: [BOI] + offset_tokens + [EOI]
    img_segment = [BOI_TOKEN] + offset_image_tokens(image_tokens) + [EOI_TOKEN]

    # Split text on the FIRST <image> occurrence
    parts = text.split("<image>", 1)
    before_text = parts[0]
    after_text = parts[1] if len(parts) > 1 else ""

    merged = [BOS_TOKEN]

    # Tokenize text before <image> (may be empty)
    if before_text:
        merged += tokenizer.encode(before_text).ids

    # Splice in image tokens
    merged += img_segment

    # Tokenize text after <image>
    if after_text:
        merged += tokenizer.encode(after_text).ids

    merged += [EOS_TOKEN]
    return merged


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────
def load_test_samples(
    data_dir: str,
    task: str,
) -> List[Dict[str, Any]]:
    """Load test samples from instructions test.json (preferred)
    or data/ jsonl fallback. Returns list of dicts with keys:
        question_id, image (relative path or None), text, answer
    """
    cfg = TASK_CONFIGS[task]
    primary_path = os.path.join(data_dir, cfg["instruction_test"])

    samples: List[Dict[str, Any]] = []

    if os.path.isfile(primary_path):
        with open(primary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        image_dir_override = cfg.get("image_dir_override", "")
        for obj in data:
            img_rel = obj.get("image", "")
            if img_rel:
                img_rel = img_rel.lstrip("./")
                # If image_dir_override is set, remap to <override_dir>/<basename>
                if image_dir_override:
                    img_rel = os.path.join(image_dir_override, os.path.basename(img_rel))
            samples.append({
                "question_id": obj.get("question_id", len(samples)),
                "image": img_rel if img_rel else None,
                "text": obj.get("text", ""),
                "answer": obj.get("answer", ""),
            })
        print(f"[{task}] Loaded {len(samples)} samples from {primary_path}")
        return samples

    # Fallback: data/ jsonl
    fallback_key = "data_fallback" if "data_fallback" in cfg else "data_test"
    fallback_path = os.path.join(data_dir, cfg[fallback_key])
    if os.path.isfile(fallback_path):
        import re

        with open(fallback_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                raw_img = obj.get("image_path", "")
                if raw_img:
                    raw_img = raw_img.replace("\\", "")
                    m = re.search(r"(ScienceQA|TextVQA|GQA)/(.*)", raw_img)
                    img_rel = m.group(0) if m else raw_img
                else:
                    img_rel = None
                samples.append({
                    "question_id": obj.get("question_id", len(samples)),
                    "image": img_rel,
                    "text": obj.get("text", ""),
                    "answer": obj.get("answer", ""),
                })
        print(f"[{task}] Loaded {len(samples)} samples from fallback {fallback_path}")
    else:
        raise FileNotFoundError(
            f"Cannot find test data for {task}: tried {primary_path} and {fallback_path}"
        )
    return samples


# ──────────────────────────────────────────────────────────────────────
# Batched VQGAN encoding
# ──────────────────────────────────────────────────────────────────────
def batch_tokenize_images(
    image_paths: List[str],
    image_tokenizer: ImageTokenizer,
    batch_size: int = 16,
    device: str = "cuda",
) -> Dict[str, List[int]]:
    """
    Encode every image through the VQGAN in batches.
    Returns {image_path: [1024 codebook indices]}.
    """
    results: Dict[str, List[int]] = {}
    unique_paths = list(dict.fromkeys(image_paths))  # deduplicate, keep order

    for start in tqdm(range(0, len(unique_paths), batch_size), desc="  VQGAN encode"):
        batch_paths = unique_paths[start : start + batch_size]
        tensors = []
        valid_paths = []
        for p in batch_paths:
            try:
                img = PIL.Image.open(p).convert("RGB")
                t = preprocess_image(img)
                tensors.append(t)
                valid_paths.append(p)
            except Exception as e:
                print(f"  [WARN] Failed to load image {p}: {e}")
                # Use a blank image as fallback
                blank = PIL.Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
                t = preprocess_image(blank)
                tensors.append(t)
                valid_paths.append(p)

        if not tensors:
            continue

        batch_tensor = torch.stack(tensors).to(device)
        with torch.no_grad():
            img_toks = image_tokenizer.image_token_from_tensor(batch_tensor)
            img_toks = img_toks.cpu().view(len(tensors), -1)

        for path, toks in zip(valid_paths, img_toks):
            results[path] = toks.tolist()

    return results


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def tokenize_task(
    task: str,
    args: argparse.Namespace,
    image_tokenizer: ImageTokenizer,
    text_tokenizer: Tokenizer,
) -> None:
    cfg = TASK_CONFIGS[task]
    output_dir = os.path.join(args.output_dir, cfg["output_name"])
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "test_data_tokenized.jsonl")

    if os.path.isfile(output_path) and not args.overwrite:
        print(f"[{task}] Output already exists: {output_path} (use --overwrite to redo)")
        return

    # 1. Load samples
    samples = load_test_samples(args.data_dir, task)
    if not samples:
        print(f"[{task}] No samples found, skipping.")
        return

    # 2. Collect unique image paths that need tokenizing
    img_paths_to_encode: List[str] = []
    for s in samples:
        if s["image"]:
            abs_path = os.path.join(args.image_base_dir, s["image"])
            img_paths_to_encode.append(abs_path)

    # 3. Batch VQGAN encode
    img_token_map: Dict[str, List[int]] = {}
    if img_paths_to_encode:
        print(f"[{task}] Tokenizing {len(set(img_paths_to_encode))} unique images ...")
        img_token_map = batch_tokenize_images(
            img_paths_to_encode,
            image_tokenizer,
            batch_size=args.batch_size,
            device=args.device,
        )

    # 4. Build final token sequences and write JSONL
    print(f"[{task}] Building token sequences and writing to {output_path} ...")
    with open(output_path, "w", encoding="utf-8") as out_f:
        for s in tqdm(samples, desc=f"  [{task}] tokenize"):
            image_tokens = None
            abs_img_path = None
            if s["image"]:
                abs_img_path = os.path.join(args.image_base_dir, s["image"])
                image_tokens = img_token_map.get(abs_img_path)

            # Ensure <image> placeholder in text if there's an image
            text = s["text"]
            if image_tokens is not None and "<image>" not in text:
                text = "<image>\n" + text

            tokens = build_full_token_sequence(text, image_tokens, text_tokenizer)

            record = {
                "question_id": s["question_id"],
                "image_path": abs_img_path or "",
                "image_tokens": image_tokens if image_tokens is not None else [],
                "text": text,
                "answer": s["answer"],
                "tokens": tokens,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[{task}] Done. Wrote {len(samples)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize test-set images for continual learning evaluation "
                    "(ScienceQA / TextVQA / GQA) using Anole's VQGAN."
    )
    parser.add_argument(
        "--anole_dir", type=str,
        default="/data1/data/kangborui/zhongyukun/medmax/Anole-7b-v0.1",
        help="Path to Anole-7b-v0.1 directory (contains tokenizer/ subfolder).",
    )
    parser.add_argument(
        "--data_dir", type=str,
        default="/data1/data/kangborui/zhongyukun/medmax/MoDE-official",
        help="Path to MoDE-official directory.",
    )
    parser.add_argument(
        "--image_base_dir", type=str,
        default="/data1/data/kangborui/zhongyukun/medmax/coin_raw",
        help="Base directory containing raw images (ScienceQA/, TextVQA/, GQA/ subdirs).",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="/data1/data/kangborui/zhongyukun/medmax/MoDE-official/data",
        help="Base directory for output JSONL files.",
    )
    parser.add_argument(
        "--tasks", nargs="+",
        default=["scienceqa", "textvqa", "gqa"],
        choices=["scienceqa", "textvqa", "gqa", "imagenet"],
        help="Which tasks to process.",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="VQGAN batch size.")
    parser.add_argument("--device", type=str, default="cuda", help="Device for VQGAN.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output.")

    args = parser.parse_args()

    # ── Load Anole VQGAN + text tokenizer ─────────────────────────────
    anole_dir = Path(args.anole_dir)
    vqgan_cfg = (anole_dir / "tokenizer" / "vqgan.yaml").as_posix()
    vqgan_ckpt = (anole_dir / "tokenizer" / "vqgan.ckpt").as_posix()
    text_tok_path = (anole_dir / "tokenizer" / "text_tokenizer.json").as_posix()

    print(f"Loading VQGAN from {vqgan_ckpt} ...")
    image_tokenizer = ImageTokenizer(
        cfg_path=vqgan_cfg,
        ckpt_path=vqgan_ckpt,
        device=args.device,
    )

    print(f"Loading text tokenizer from {text_tok_path} ...")
    text_tokenizer = Tokenizer.from_file(text_tok_path)

    # ── Process each task ─────────────────────────────────────────────
    for task in args.tasks:
        print(f"\n{'='*60}")
        print(f"Processing task: {task}")
        print(f"{'='*60}")
        tokenize_task(task, args, image_tokenizer, text_tokenizer)

    print("\nAll done!")


if __name__ == "__main__":
    main()
