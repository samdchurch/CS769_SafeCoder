import os
import random
import pickle
import torch
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModelForCausalLM
from time import time

# Constants and imports specific to the project
from .constants import FUNC, GOOD, BAD, PRETRAINED_MODELS, CHAT_MODELS

class Timer:
    """
    Timer utility to track progress in repetitive tasks, such as training loops.
    """

    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.total_time_elapsed = 0.0
        self.recorded_steps = 0
        self.running_avg = None
        self.last_measured_time = None

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
        remaining = self.total_steps - self.recorded_steps
        self.completion = self.recorded_steps / self.total_steps
        est_time = remaining * self.running_avg
        self.remaining_hours, self.remaining_minutes, self.remaining_seconds = self._convert_seconds_to_h_m_s(est_time)

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
        with open(f"data_{mode}.pkl", "rb") as f:
            self.dataset = pickle.load(f)
        self.args.logger.info("***** Loaded dataset *****")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return tuple(torch.tensor(t) for t in self.dataset[idx])

class LossDict:
    def __init__(self, keys):
        self.d = {key: [] for key in keys}

    def step(self, other):
        for k in other.d:
            self.d[k].extend(other.d[k])

    def pretty_print(self, args):
        return ", ".join(f"{k}: {round(sum(l) / len(l) / args.grad_acc_steps, 6)}" for k, l in self.d.items() if l)

    def clear(self):
        for key in self.d:
            self.d[key].clear()

def token_weighted_loss(loss_type, inputs, targets, weights):
    inputs, targets, weights = inputs.view(-1, inputs.size(-1)), targets.view(-1), weights.view(-1)
    loss_fct = {'ce': torch.nn.CrossEntropyLoss(reduction="none"),
                'nll': torch.nn.NLLLoss(reduction="none"),
                'kl': torch.nn.KLDivLoss(log_target=True, reduction="none")}
    loss = loss_fct[loss_type](inputs, targets)
    loss = loss[weights != 0]
    return loss.mean()

def get_logits_from_lm(lm, inputs, control_ids=None):
    past = lm.get_past_from_prefix(control_ids) if control_ids is not None else None
    outputs = lm(inputs, past_key_values=past)
    shift_logits, shift_labels = outputs.logits[..., :-1, :], inputs[..., 1:].unsqueeze(-1)
    shift_probs = F.softmax(shift_logits, dim=-1)
    return shift_logits.squeeze(0), torch.gather(shift_probs, 2, shift_labels).squeeze(-1).squeeze(0)

class Trainer:
    def __init__(self, args):
        self.args = args
        self.loss_keys = ["func", "pos", "neg"]
        if self.args.kl_loss_weight > 0:
            self.loss_keys.append("kl")

    def step(self, batch):
        loss_dict = LossDict(self.loss_keys)
        sample_types, inputs, weights = batch
        inputs, weights = inputs.to(self.model.device), weights.to(self.model.device)
        shift_inputs, shift_weights = inputs[..., 1:], weights[..., 1:]
        outputs = self.model(inputs)
        shift_logits = outputs.logits[..., :-1, :]

        loss_total = 0.0
        for sample_type in sample_types:
            if sample_type in (FUNC, GOOD):
                loss = token_weighted_loss("ce", shift_logits, shift_inputs, shift_weights) * (self.args.loss_weight if sample_type == GOOD else 1)
                loss_dict[sample_type].append(loss.item())
                loss_total += loss
            elif sample_type == BAD:
                loss = token_weighted_loss("ul", shift_logits, shift_inputs, shift_weights) * self.args.loss_weight
                loss_dict["neg"].append(loss.item())
                loss_total += loss

            if sample_type in (GOOD, BAD) and self.args.kl_loss_weight > 0:
                ref_outputs = self.ref_model(inputs).logits[..., :-1, :]
                kl_loss = token_weighted_loss("kl", F.log_softmax(shift_logits, dim=-1), F.log_softmax(ref_outputs, dim=-1), 1 - shift_weights) * self.args.kl_loss_weight / 1000
                loss_dict["kl"].append(kl_loss.item())
                loss_total += kl_loss

        return loss_total, loss_dict

    def do_eval(self):
        val_dataloader = DataLoader(self.val_dataset, sampler=SequentialSampler(self.val_dataset), batch_size=1)
        acc_loss_dict = LossDict(self.loss_keys)
        for batch in val_dataloader:
            loss, loss_dict = self.step(batch)
            acc_loss_dict.step(loss_dict)
        return acc_loss_dict.pretty_print(self.args)

    def set_seed_util(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def load_model(self):
        model_name = self.args.pretrain_name
        model_dir = PRETRAINED_MODELS.get(model_name) or CHAT_MODELS.get(model_name) or os.path.join(self.args.model_dir, model_name, "checkpoint-last")
        assert os.path.exists(model_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto")
        self.model.resize_token_embeddings(len(self.tokenizer))
        if self.args.kl_loss_weight > 0 and not self.args.sven:
            self.ref_model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto")
            self.ref_model.eval()

    def load_dataset(self):
        self.dataset = CodeDataset(self.args, self.tokenizer, "train")
        self.val_dataset = CodeDataset(self.args, self.tokenizer, "val")

    def save(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def run(self):
        self.load_model()
        self.load_dataset()
        timer = Timer(self.args.num_train_epochs * len(self.dataset) // self.args.batch_size)
        timer.start()
        
        optimizer, scheduler = self.configure_optimizer_scheduler()
        global_step = 0
        acc_loss_dict = LossDict(self.loss_keys)

        self.model.train()
        for epoch in range(self.args.num_train_epochs):
            for step, batch in enumerate(DataLoader(self.dataset, sampler=RandomSampler(self.dataset), batch_size=self.args.batch_size)):
                loss, loss_dict = self.step(batch)
                (loss / self.args.grad_acc_steps).backward()
                acc_loss_dict.step(loss_dict)
                
                if (step + 1) % self.args.grad_acc_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % self.args.logging_steps == 0:
                        self.args.logger.info(f"Epoch: {epoch+1}, Step: {global_step}, Loss: {acc_loss_dict.pretty_print(self.args)}, Timer: {timer}")
                        acc_loss_dict.clear()

                    timer.end()
                    timer.start()

            if (epoch + 1) % self.args.save_epochs == 0:
                self.evaluate_and_save(epoch)

        if self.args.save_epochs == 0 or (epoch + 1) % self.args.save_epochs != 0:
            self.evaluate_and_save("final")

    def configure_optimizer_scheduler(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad], "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=self.args.num_train_epochs * len(self.dataset) // self.args.batch_size)
        return optimizer, scheduler

    def evaluate_and_save(self, epoch):
        self.model.eval()
        eval_loss = self.do_eval()
        self.args.logger.info(f"Eval loss after epoch {epoch}: {eval_loss}")
        save_path = os.path.join(self.args.output_dir, f"checkpoint-epoch-{epoch}")
        self.save(save_path)
