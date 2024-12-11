import argparse
import json
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def create_output_file(output_file):
    # Check if the output directory exists, create it if not
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # If the output file doesn't exist, it will be created when opening in write mode
    if not os.path.exists(output_file):
        open(output_file, 'w').close()

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files", nargs="+", required=True, help="Paths to the input JSONL files")
    parser.add_argument("--output_dir", required=True, help="Directory to save the processed output JSONL files")
    parser.add_argument("--model_dir_weak", required=True, help="Directory of the weak model")
    parser.add_argument("--model_dir_sft", required=True, help="Directory of the fine-tuned model")
    parser.add_argument("--beta", type=float, default=1.0, help="Scaling factor for reward computation")
    return parser.parse_args()

def load_models_and_tokenizers(model_dir_weak, model_dir_sft, device):
    tokenizer_weak = AutoTokenizer.from_pretrained(model_dir_weak)
    model_weak = AutoModelForCausalLM.from_pretrained(model_dir_weak).to(device)
    tokenizer_sft = AutoTokenizer.from_pretrained(model_dir_sft)
    model_sft = AutoModelForCausalLM.from_pretrained(model_dir_sft).to(device)
    return tokenizer_weak, model_weak, tokenizer_sft, model_sft

def compute_dpo_reward(prompt, response, model_weak, model_sft, tokenizer, beta, device):
    inputs = tokenizer(prompt + response, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs_weak = model_weak(**inputs)
        logits_weak = outputs_weak.logits
    log_probs_weak = torch.log_softmax(logits_weak, dim=-1)
    log_prob_weak = sum(log_probs_weak[0, i, token_id].item() for i, token_id in enumerate(input_ids[0]))

    with torch.no_grad():
        outputs_sft = model_sft(**inputs)
        logits_sft = outputs_sft.logits
    log_probs_sft = torch.log_softmax(logits_sft, dim=-1)
    log_prob_sft = sum(log_probs_sft[0, i, token_id].item() for i, token_id in enumerate(input_ids[0]))

    reward = beta * (log_prob_weak - log_prob_sft)
    return reward

def process_file(input_file, output_file, tokenizer_weak, model_weak, tokenizer_sft, model_sft, beta, device):
    create_output_file(output_file)
    number_of_swaps = 0

    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in tqdm(infile, desc=f"Processing {input_file}"):
            try:
                data = json.loads(line)
                prompt = data["description"]
                before = data["func_src_before"]
                after = data["func_src_after"]

                reward_before = compute_dpo_reward(prompt, before, model_weak, model_sft, tokenizer_weak, beta, device)
                reward_after = compute_dpo_reward(prompt, after, model_weak, model_sft, tokenizer_weak, beta, device)

                if reward_after < reward_before:
                    data["func_src_before"], data["func_src_after"] = data["func_src_after"], data["func_src_before"]
                    number_of_swaps += 1

                outfile.write(json.dumps(data) + "\n")
            except Exception as e:
                print(f"Error processing line in {input_file}: {e}")
        print(f"number of swaps {number_of_swaps}")

def main():
    args = parse_arguments()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer_weak, model_weak, tokenizer_sft, model_sft = load_models_and_tokenizers(
        args.model_dir_weak, args.model_dir_sft, device
    )

    for input_file in args.input_files:
        output_file = f"{args.output_dir}/{input_file.split('/')[-1]}"
        print(f"Processing file: {input_file} -> {output_file}")
        process_file(
            input_file,
            output_file,
            tokenizer_weak,
            model_weak,
            tokenizer_sft,
            model_sft,
            args.beta,
            device
        )

if __name__ == "__main__":
    main()
