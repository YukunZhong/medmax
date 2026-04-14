import argparse
import os

import torch
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model
from transformers import ChameleonForConditionalGeneration

def main(args):
    adapter_path = args.ckpt_path
    base_path = args.base_path

    base_model = ChameleonForConditionalGeneration.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if os.path.isfile(os.path.join(adapter_path, "adapter_config.json")):
        _ = PeftConfig.from_pretrained(adapter_path)
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        adapter_file = adapter_path
        if os.path.isdir(adapter_path):
            adapter_file = os.path.join(adapter_path, "mode_adapters.pt")
        state_dict = torch.load(adapter_file, map_location="cpu")

        peft_config = LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, peft_config)
        model.load_state_dict(state_dict, strict=False)

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