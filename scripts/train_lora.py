import os
import argparse

from safecoder.utils import set_seed, set_logging
from safecoder.trainer import Trainer

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_name', type=str, required=True)
    parser.add_argument('--datasets', type=str, nargs='+', required=True)
    parser.add_argument('--pretrain_name', type=str, default='codegen-350m')

    # Training arguments
    parser.add_argument('--loss_weight', type=float, default=1.0)
    parser.add_argument('--sven', action='store_true', default=False)
    parser.add_argument('--num_train_epochs', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--max_num_tokens', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--grad_acc_steps', type=int, default=32)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--kl_loss_weight', type=int, default=0)
    parser.add_argument('--exclude_neg', action='store_true', default=False)
    parser.add_argument('--no_weights', action='store_true', default=False)

    # LoRA arguments
    parser.add_argument('--lora', action='store_true', default=False, help='Enable LoRA fine-tuning')
    parser.add_argument('--r', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha parameter')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='Dropout rate for LoRA layers')

    # Upsampling arguments
    parser.add_argument('--sampling_size', type=int, default=-1)
    parser.add_argument('--sampling_method', type=str, choices=['uniform', 'inverse-prop', 'minority'], default='minority')
    parser.add_argument('--cwes', type=str, nargs='*', default=['all'])
    parser.add_argument('--langs', type=str, nargs='*', default=['all'])

    parser.add_argument('--logging_steps', type=int, default=50)
    parser.add_argument('--save_epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--data_dir', type=str, default='../data_train_val')
    parser.add_argument('--model_dir', type=str, default='../trained/')

    args = parser.parse_args()

    # Adjust naming convention based on flags
    if args.lora and not args.output_name.startswith(f'{args.pretrain_name}-lora'):
        args.output_name = f'{args.pretrain_name}-lora-' + args.output_name

    if args.sven and not args.output_name.startswith(f'{args.pretrain_name}-sven'):
        args.output_name = f'{args.pretrain_name}-sven-' + args.output_name

    if args.sampling_size == -1 and 'lmsys' in args.datasets:
        args.sampling_size = 40

    if args.sampling_size == -1 and 'evol' in args.datasets:
        args.sampling_size = 20

    if args.num_train_epochs is None:
        args.num_train_epochs = 5 if args.sven else (5 if args.pretrain_name.startswith('codellama') else 2)

    if args.learning_rate is None:
        args.learning_rate = 5e-2 if args.sven else (1e-3 if args.pretrain_name.startswith('codellama') else 2e-5)

    if args.exclude_neg:
        args.sampling_size = args.sampling_size // 2

    # Create output directory
    args.output_dir = os.path.join(args.model_dir, args.output_name)
    os.makedirs(args.output_dir, exist_ok=True)

    return args

def main():
    args = get_args()
    set_logging(args, os.path.join(args.output_dir, 'train.log'))
    set_seed(args.seed)
    Trainer(args).run()

if __name__ == '__main__':
    main()