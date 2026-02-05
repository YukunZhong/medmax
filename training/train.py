import os
import json
import torch
import wandb
import deepspeed
import argparse

from transformers import ChameleonForConditionalGeneration, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from .data import TokenizedDataset, collate_fn, create_new_tokens
from .parser import parse_args

def main():

    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    if args.wandb:
        if local_rank == 0:
            wandb.init(
                entity=args.wandb_entity,
                project="chlm",
                config=args
            )
            if args.name != "":
                wandb.run.name = args.name
    else:
        os.environ['WANDB_DISABLED'] = 'true'

    os.makedirs(args.output_dir, exist_ok = True)
    
    # Initialize the model
    model = ChameleonForConditionalGeneration.from_pretrained(args.ckpt)
    if args.extend_vocab:
        model.resize_token_embeddings(len(tokenizer))

    if args.lora:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj","gate_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
    # Initialize the dataset
    train_dataset = TokenizedDataset(args.train_data, max_length=args.max_length)
    val_dataset = TokenizedDataset(args.val_data, max_length=args.max_length) if args.val_data != "" else None

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epoch,
        max_steps=args.steps,
        gradient_accumulation_steps=args.grad_acc,
        per_device_train_batch_size=args.bs,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        eval_strategy="steps" if args.val_data != "" else "no",
        eval_steps=args.eval_steps,
        deepspeed=args.ds,
        report_to="wandb" if args.wandb else "none",
    )

    # Initialize the Trainer with custom collate_fn
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn
    )

    # Train the model
    trainer.train(resume_from_checkpoint=args.resume)
        
if __name__ == "__main__":
    main()