import argparse
import os

import torch
from peft import PeftConfig, PeftModel
from transformers import ChameleonForConditionalGeneration

def main(args):
    adapter_path = args.ckpt_path
    base_path = args.base_path

    # Use the adapter's own config instead of rebuilding LoRA hyper-parameters by hand.
    # This avoids rank mismatch errors (e.g. ckpt r=8 vs manually-set r=16).
    _ = PeftConfig.from_pretrained(adapter_path)
    base_model = ChameleonForConditionalGeneration.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)

    model = model.merge_and_unload()
    os.makedirs(args.output_dir, exist_ok = True)

    model.save_pretrained(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", help = 'path to trained lora checkpoint')
    parser.add_argument("--base_path", help = 'path to base path')
    parser.add_argument("--output_dir", help = 'path to output dir')
    args = parser.parse_args()
    main(args)