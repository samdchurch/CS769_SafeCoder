import os
import re
import torch
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, LoraConfig, TaskType
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
import random

# from .dataset import CodeDataset
from .constants import FUNC, GOOD, BAD
from .constants import PRETRAINED_MODELS, CHAT_MODELS

from time import time
import pickle


class Timer:

    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.total_time_elapsed = 0.0
        self.recorded_steps = 0
        self.running_avg = None
        self.last_measured_time = None
        self.completion = 0.0

    def __str__(self):
        self._calculate_completion_and_time_remaining()
        return f"{int(self.completion * 100)}% complete: {self.remaining_hours}h {self.remaining_minutes}m {self.remaining_seconds}s"

    def start(self):
        self.last_measured_time = time()

    def end(self):
        elapsed = time() - self.last_measured_time
        self.total_time_elapsed += elapsed
        self.recorded_steps += 1
        self.running_avg = elapsed if self.running_avg is None else (self.running_avg * (self.recorded_steps - 1) + elapsed) / self.recorded_steps

    def _calculate_completion_and_time_remaining(self):
        remaining_steps = self.total_steps - self.recorded_steps
        self.completion = self.recorded_steps / self.total_steps if self.total_steps > 0 else 0
        estimated_time_remaining = remaining_steps * self.running_avg if self.running_avg else 0
        self.remaining_hours, self.remaining_minutes, self.remaining_seconds = self._convert_seconds_to_h_m_s(estimated_time_remaining)

    @staticmethod
    def _convert_seconds_to_h_m_s(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return hours, minutes, seconds

    def duration(self):
        hours, minutes, seconds = self._convert_seconds_to_h_m_s(self.total_time_elapsed)
        print(f"Completed in {hours}h {minutes}m {seconds}s")


class CodeDataset(Dataset):
    def __init__(self, args, tokenizer, mode):
        self.args = args
        with open(f'data_{mode}.pkl', 'rb') as f:
            loaded_data = pickle.load(f)
        self.dataset = loaded_data
        self.args.logger.info('***** saved dataset *****')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return tuple(torch.tensor(t) for t in self.dataset[item])


class LossDict:
    def __init__(self, keys):
        self.loss_data = {key: [] for key in keys}

    def step(self, other):
        for key, values in other.loss_data.items():
            self.loss_data[key].extend(values)

    def pretty_print(self, args):
        return ", ".join(f"{key}: {round(sum(values) / len(values) / args.grad_acc_steps, 6)}" 
                         for key, values in self.loss_data.items() if values)

    def clear(self):
        for key in self.loss_data:
            self.loss_data[key].clear()

    def __getitem__(self, key):
        return self.loss_data[key]

def token_weighted_loss(loss_type, inputs, targets, weights):
    inputs = inputs.view(-1, inputs.size(-1))
    targets = targets.view(-1)
    weights = weights.view(-1)

    if loss_type == 'ce':
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(inputs, targets)
        
    elif loss_type == 'nll':
        loss_fct = torch.nn.NLLLoss(reduction='none')
        loss = loss_fct(inputs, targets)
        
    elif loss_type == 'ul':
        # Ensure `probs` has the correct dimensionality
        probs = F.softmax(inputs, dim=-1)
        if probs.dim() == 2:
            probs = probs.unsqueeze(1)  # Add a singleton dimension if probs is 2D
        gathered_probs = torch.gather(probs, 2, targets.view(-1, 1, 1)).squeeze(-1)
        loss = -torch.log(torch.clamp(1.0 - gathered_probs, min=1e-5))
        
    elif loss_type == 'kl':
        targets = targets.view(-1, targets.size(-1))  # Match dimensions for KL divergence
        loss_fct = torch.nn.KLDivLoss(log_target=True, reduction='none')
        loss = loss_fct(inputs, targets).sum(dim=1)
        
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

    loss = loss[weights != 0]
    return loss.mean()


class Trainer:
    def __init__(self, args):
        self.args = args
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.loss_keys = ['func', 'pos', 'neg']
        if self.args.kl_loss_weight > 0:
            self.loss_keys.append('kl')

    def load_model_util(self, model_name, args):
        # Original logic to handle model loading from PRETRAINED_MODELS, CHAT_MODELS, or Hugging Face
        if model_name in PRETRAINED_MODELS:
            model_dir = PRETRAINED_MODELS[model_name]
        elif model_name in CHAT_MODELS:
            model_dir = CHAT_MODELS[model_name]
        else:
            if 'checkpoint-epoch' in model_name:
                model_dir = os.path.join(args.model_dir, model_name)
            else:
                model_dir = os.path.join(args.model_dir, model_name, 'checkpoint-last')
            assert os.path.exists(model_dir)

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map='auto',
            trust_remote_code=True
        )
        model.resize_token_embeddings(len(tokenizer))
        return tokenizer, model

    def load_model(self):
        # Load base model and tokenizer
        self.tokenizer, self.model = self.load_model_util(self.args.pretrain_name, self.args)
        self.model.train()
        print("here------------------------------")
        print(self.model)
        # If LoRA is enabled, modify the model with LoRA configurations
        if self.args.lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.args.r,
                lora_alpha=self.args.lora_alpha,
                lora_dropout=self.args.lora_dropout,
                target_modules=["attn.c_proj"]  # Focus on output projection
            )


            self.model = get_peft_model(self.model, lora_config)
            self.args.logger.info(f"LoRA model initialized with config: {lora_config}")

        # Reference model for KL divergence if required
        if self.args.kl_loss_weight > 0 and not self.args.sven:
            _, self.ref_model = self.load_model_util(self.args.pretrain_name, self.args)
            self.ref_model.eval()
    def step(self, batch):
        loss_dict = LossDict(self.loss_keys)
        sample_types, inputs, weights = batch

        inputs = inputs.to(self.model.device)
        shift_inputs = inputs[..., 1:]
        weights = weights.to(self.model.device)
        shift_weights = weights[..., 1:]
        
        outputs = self.model(inputs)
        shift_logits = outputs.logits[..., :-1, :]

        total_loss = 0.0
        for sample_type in sample_types:
            total_loss += self._compute_sample_loss(sample_type, shift_logits, shift_inputs, shift_weights, loss_dict)
            
            if sample_type in {GOOD, BAD} and self.args.kl_loss_weight > 0:
                total_loss += self._compute_kl_loss(shift_logits, inputs, shift_weights, loss_dict)
                
        return total_loss, loss_dict

    def _compute_sample_loss(self, sample_type, logits, inputs, weights, loss_dict):
        if sample_type == FUNC:
            loss = token_weighted_loss("ce", logits, inputs, weights)
            loss_dict["func"].append(loss.item())
        elif sample_type == GOOD:
            loss = self.args.loss_weight * token_weighted_loss("ce", logits, inputs, weights)
            loss_dict["pos"].append(loss.item())
        elif sample_type == BAD:
            loss = self.args.loss_weight * token_weighted_loss("ul", logits, inputs, weights)
            loss_dict["neg"].append(loss.item())
        else:
            raise ValueError("Unknown sample type.")
        
        return loss

    def _compute_kl_loss(self, logits, inputs, weights, loss_dict):
        with torch.no_grad():
            ref_outputs = self.ref_model(inputs)
        
        shift_ref_log_probs = F.log_softmax(ref_outputs.logits[..., :-1, :], dim=-1)
        shift_log_probs = F.log_softmax(logits, dim=-1)
        
        kl_loss = self.args.kl_loss_weight * token_weighted_loss("kl", shift_log_probs, shift_ref_log_probs, 1 - weights) / 1000
        loss_dict["kl"].append(kl_loss.item())
        
        return kl_loss

    def do_eval(self):
        val_sampler = SequentialSampler(self.val_dataset)
        val_dataloader = DataLoader(self.val_dataset, sampler=val_sampler, batch_size=1)
        acc_loss_dict = LossDict(self.loss_keys)
        for batch in val_dataloader:
            # loss, loss_dict = self.sven_step(batch) if self.args.sven else 
            loss, loss_dict = self.step(batch)
            acc_loss_dict.step(loss_dict)
        return acc_loss_dict.pretty_print(self.args)
    def load_dataset(self):
        self.dataset = CodeDataset(self.args, self.tokenizer, 'train')
        self.val_dataset = CodeDataset(self.args, self.tokenizer, 'val')
    def set_seed_util(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    def _configure_optimizer(self):
        if self.args.lora:
            # Only optimize LoRA parameters
            parameters = [p for p in self.model.parameters() if p.requires_grad]
        else:
            # Standard optimizer logic for the entire model
            no_decay = ["bias", "LayerNorm.weight"]
            parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
                    "weight_decay": self.args.weight_decay
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
                    "weight_decay": 0.0
                }
            ]
        return AdamW(parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)

    def run(self):
        # Load the model and dataset
        self.load_model()
        self.load_dataset()

        # Prepare dataloader
        train_dataloader = DataLoader(
            self.dataset,
            sampler=RandomSampler(self.dataset),
            batch_size=self.args.batch_size,
            drop_last=True
        )

        # Configure optimizer and scheduler
        optimizer = self._configure_optimizer()
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=(len(train_dataloader) * self.args.num_train_epochs)
        )

        # Training loop
        global_step, acc_loss_dict = 0, LossDict(self.loss_keys)
        self.set_seed_util(self.args.seed)
        timer = Timer(total_steps=(len(train_dataloader) * self.args.num_train_epochs))
        timer.start()

        for epoch in range(self.args.num_train_epochs):
            self.model.train()
            for step, batch in enumerate(train_dataloader):
                loss, loss_dict = self.step(batch)
                loss /= self.args.grad_acc_steps
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                acc_loss_dict.step(loss_dict)

                if (step + 1) % self.args.grad_acc_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1

                    if global_step % self.args.logging_steps == 0:
                        acc_loss_pp = acc_loss_dict.pretty_print(self.args)
                        self.args.logger.info(
                            f"Epoch: {epoch + 1}/{self.args.num_train_epochs}, "
                            f"Step: {global_step}, Loss: {acc_loss_pp}, Timer: {timer}"
                        )
                        acc_loss_dict.clear()

                    timer.end()
                    timer.start()

            # Save checkpoint after each epoch
            self._save_checkpoint(epoch + 1)

        # Final checkpoint
        self._save_final_checkpoint()

