import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
import argparse

# ============================
# Data Processing Script
# ============================
class PreferenceDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length=512):
        self.data = self.load_data(data_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_data(self, data_file):
        with open(data_file, 'r') as f:
            return [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        prompt = sample["prompt"]
        preferred_output = sample["preferred_output"]
        less_preferred_output = sample["less_preferred_output"]

        prompt_ids = self.tokenizer(prompt, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        preferred_ids = self.tokenizer(preferred_output, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        less_preferred_ids = self.tokenizer(less_preferred_output, truncation=True, padding="max_length",  max_length=self.max_length, return_tensors="pt")

        return (
            prompt_ids["input_ids"].squeeze(0),
            preferred_ids["input_ids"].squeeze(0),
            less_preferred_ids["input_ids"].squeeze(0),
        )

# ============================
# Training Script
# ============================
def train_model(model_name, data_file, output_dir, num_epochs=3, batch_size=8, learning_rate=5e-5, max_length=512):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = PreferenceDataset(data_file, tokenizer, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    def dpo_loss(logits_preferred, logits_less_preferred):
        diff = logits_preferred - logits_less_preferred
        return -torch.log(torch.sigmoid(diff)).mean()

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            prompt_ids, preferred_ids, less_preferred_ids = [x.to(device) for x in batch]

            preferred_outputs = model(input_ids=prompt_ids, labels=preferred_ids)
            less_preferred_outputs = model(input_ids=prompt_ids, labels=less_preferred_ids)

            preferred_logits = preferred_outputs.logits.mean(dim=1)
            less_preferred_logits = less_preferred_outputs.logits.mean(dim=1)

            loss = dpo_loss(preferred_logits, less_preferred_logits)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")

    # Save the fine-tuned model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a language model using DPO optimization.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the pretrained model to use.")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the processed JSONL dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the fine-tuned model.")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length for tokenization.")

    args = parser.parse_args()

    train_model(
        model_name=args.model_name,
        data_file=args.data_file,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length
    )
