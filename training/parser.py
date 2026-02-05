import argparse

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--train_data', required=True, type=str, help="path to tokenized train data")
    parser.add_argument('--val_data', default="", type=str, help="path to tokenized val data")

    parser.add_argument('--ckpt', required=True, type=str, help="finetuning checkpoint")
    parser.add_argument('--ds', required=True, type=str, help="deepspeed config")
    parser.add_argument('--output_dir', required=True, type=str, help="output directory")

    parser.add_argument('--resume', default=False, action='store_true', help='resume from the last checkpoint')         
    parser.add_argument('--extend_vocab', default=False, action='store_true', help='extend vocabulary embeddings')
    
    parser.add_argument('--lora', default=False, action='store_true', help='lora finetuning')         
    parser.add_argument('--lora_r', type=int, default=16, help="lora r")         
    parser.add_argument('--lora_alpha', type=int, default=16, help="lora alpha")         
    parser.add_argument('--lora_dropout', type=float, default=0.05, help="lora dropout")         

    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate")         
    parser.add_argument('--epoch', type=int, default=1, help="num of epochs")         
    parser.add_argument('--grad_acc', type=int, default=1, help="gradient accumulation steps")         
    parser.add_argument('--steps', type=int, default=-1, help="num of training steps")         
    parser.add_argument('--bs', type=int, default=1, help="per device batch size")  
    parser.add_argument('--max_length', type=int, default=2048, help="maximum token length per sample")         
    parser.add_argument('--save_strategy', default="no", type=str, choices=["no", "epoch", "steps"], help="save strategy")
    parser.add_argument('--save_steps', type=float, default=0.1, help="save checkpoints every save_steps of the training run")         
    parser.add_argument('--logging_steps', type=float, default=0.001, help="logging at every logging_steps of the total training steps")         
    parser.add_argument('--eval_steps', type=float, default=0.1, help="evaluate at every evaluation steps of the total training steps")         
    parser.add_argument('--warmup_ratio', type=float, default=0.0, help="warmup ratio")         
    parser.add_argument('--lr_scheduler', default="cosine", type=str, help="lr scheduler")

    parser.add_argument('--bf16', default=False, action='store_true', help='bf16')         
    parser.add_argument('--fp16', default=False, action='store_true', help='fp16')         

    parser.add_argument('--wandb', default=False, action='store_true', help='wandb or not')         
    parser.add_argument('--wandb_entity', default="", type=str, help='wandb entity')         
    parser.add_argument('--name', default="", type=str, help="wandb experiment name")

    args = parser.parse_args()
    return args