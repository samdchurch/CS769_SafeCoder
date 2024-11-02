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
    """
    An object that keeps track of our progress in some repetitive loop and outputs a time estimate of the remaining time
    we will need to finish our loop. It is a handy tool for line and or grid searches, or sequential monte carlo
    simulations. Note that if the constituting steps in the loop(s) take vastly different times, the time estimate can
    be arbitrarily off, however the overall progress will still be displayed.
    """

    def __init__(self, total_steps):
        """
        Constructor.

        :param total_steps: (int) The total number of steps the measured process will make.
        """
        self.total_steps = total_steps
        self.total_time_elapsed = 0.
        self.recorded_steps = 0
        self.running_avg = None
        self.last_measured_time = None

        # time estimates
        self.remaining_seconds = None
        self.remaining_minutes = None
        self.remaining_hours = None

        # completion
        self.completion = 0.

    def __str__(self):
        self._calculate_completion_and_time_remaining()
        spaces = '          '  # to account for overhanging lines :)
        return f'{int(self.completion * 100)}%: {self.remaining_hours}h {self.remaining_minutes}m {self.remaining_seconds}s{spaces}'

    def start(self):
        """
        Mark the start of the innermost loop over which you wish to measure and record the current clock time.

        :return: None
        """
        self.last_measured_time = time()

    def end(self):
        """
        Mark the end of the innermost loop over which you wish to measure and record the current clock time. Eventually,
        update the running average estimate and add a step to the completed ones.

        :return: None
        """
        recorded_time = time() - self.last_measured_time
        self.total_time_elapsed += recorded_time
        if self.running_avg is None:
            self.running_avg = recorded_time
        else:
            self.running_avg = (self.running_avg * self.recorded_steps + recorded_time) / (self.recorded_steps + 1)
        self.recorded_steps += 1

    @staticmethod
    def _convert_seconds_to_h_m_s(seconds):
        """
        A private method to convert seconds into hours, minutes and seconds for better human readability.

        :param seconds: (int) Seconds we wish to convert into h, m, s format.
        :return: (tuple) The amount of seconds given converted into h, m, s format.
        """
        hours = int(seconds // 3600)
        minutes = int((seconds - hours * 3600) // 60)
        rem_seconds = int(seconds - hours * 3600 - minutes * 60)
        return hours, minutes, rem_seconds

    def _calculate_completion_and_time_remaining(self):
        """
        Private method to compute the completion of the process and estimate the remaining time from the running
        average.

        :return: None
        """
        remaining = self.total_steps - self.recorded_steps
        self.completion = self.recorded_steps / self.total_steps
        if self.running_avg is not None:
            estimated_time = remaining * self.running_avg
            self.remaining_hours, self.remaining_minutes, self.remaining_seconds = self._convert_seconds_to_h_m_s(estimated_time)
        else:
            self.remaining_hours = '??'
            self.remaining_minutes = '??'
            self.remaining_seconds = '??'

    def duration(self):
        """
        After the process has finished call this method to display the absolute time the completion of the whole process
        has taken.

        :return: None
        """
        h, m, s = self._convert_seconds_to_h_m_s(self.total_time_elapsed)
        print('\n')
        print(f'Completed. Time Elapsed: {h}h {m}m {s}s')


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
        self.d = OrderedDict()
        self.keys = keys
        for key in keys:
            self.d[key] = list()

    def step(self, other):
        for k in other.d:
            self.d[k] += other.d[k]

    def pretty_print(self, args):
        p = []
        for k, l in self.d.items():
            if len(l) > 0:
                s = sum(l) / len(l) / args.grad_acc_steps
                p.append(f'{k}: {round(s, 6)}')
        return ', '.join(p)

    def clear(self):
        for key in self.keys:
            self.d[key].clear()

    def __getitem__(self, k):
        return self.d[k]

def token_weighted_loss(loss_type, inputs, targets, weights):
    if loss_type == 'ce':
        inputs = inputs.view(-1, inputs.size(-1))
        targets = targets.view(-1)
        weights = weights.view(-1)
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(inputs, targets)
    elif loss_type == 'nll':
        inputs = inputs.view(-1, inputs.size(-1))
        targets = targets.view(-1)
        weights = weights.view(-1)
        loss_fct = torch.nn.NLLLoss(reduction='none')
        loss = loss_fct(inputs, targets)
    elif loss_type == 'ul':
        probs = F.softmax(inputs, dim=-1)
        probs = torch.gather(probs, 2, targets.unsqueeze(-1)).squeeze(-1)
        probs = torch.clamp((1.0-probs), min=1e-5)
        loss = -torch.log(probs)
    elif loss_type == 'kl':
        inputs = inputs.view(-1, inputs.size(-1))
        targets = targets.view(-1, targets.size(-1))
        weights = weights.view(-1)
        loss_fct = torch.nn.KLDivLoss(log_target=True, reduction='none')
        loss = loss_fct(inputs, targets)
        loss = loss.sum(dim=1)
    else:
        assert False

    loss = loss[weights != 0]
    return loss.mean()

def get_logits_from_lm(lm, inputs, control_ids):
    if control_ids is not None:
        past = lm.get_past_from_prefix(control_ids)
    else:
        past = None
    outputs = lm(inputs, past_key_values=past)
    shift_logits = outputs.logits[..., :-1, :]
    shift_labels = inputs[..., 1:].unsqueeze(-1)
    shift_probs = F.softmax(shift_logits, dim=-1)
    return shift_logits.squeeze(0), torch.gather(shift_probs, 2, shift_labels).squeeze(-1).squeeze(0)

class Trainer:
    def __init__(self, args):
        self.args = args
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.loss_keys = ['func', 'pos', 'neg']
        if self.args.kl_loss_weight > 0:
            self.loss_keys.append('kl')

    def step(self, batch):
        loss_dict = LossDict(self.loss_keys)

        sample_types, inputs, weights = batch
        inputs = inputs.to(self.model.device)
        shift_inputs = inputs[..., 1:]
        weights = weights.to(self.model.device)
        shift_weights = weights[..., 1:]
        outputs = self.model(inputs)
        shift_logits = outputs.logits[..., :-1, :]

        loss_total = 0.0
        for sample_type in sample_types:
            if sample_type == FUNC:
                loss = token_weighted_loss('ce', shift_logits, shift_inputs, shift_weights)
                loss_dict['func'].append(loss.item())
                loss_total += loss
            elif sample_type == GOOD:
                loss = self.args.loss_weight * token_weighted_loss('ce', shift_logits, shift_inputs, shift_weights)
                loss_dict['pos'].append(loss.item())
                loss_total += loss
            elif sample_type == BAD:
                loss = self.args.loss_weight * token_weighted_loss('ul', shift_logits, shift_inputs, shift_weights)
                loss_dict['neg'].append(loss.item())
                loss_total += loss
            else:
                assert False

            if (sample_type == GOOD or sample_type == BAD) and self.args.kl_loss_weight > 0:
                with torch.no_grad():
                    ref_outputs = self.ref_model(inputs)
                shift_ref_log_probs = F.log_softmax(ref_outputs.logits[..., :-1, :], dim=-1)
                shift_log_probs = F.log_softmax(shift_logits, dim=-1)
                loss = self.args.kl_loss_weight * token_weighted_loss('kl', shift_log_probs, shift_ref_log_probs, 1-shift_weights) / 1000
                loss_dict['kl'].append(loss.item())
                loss_total += loss

        return loss_total, loss_dict

    def do_eval(self):
        val_sampler = SequentialSampler(self.val_dataset)
        val_dataloader = DataLoader(self.val_dataset, sampler=val_sampler, batch_size=1)
        acc_loss_dict = LossDict(self.loss_keys)
        for batch in val_dataloader:
            # loss, loss_dict = self.sven_step(batch) if self.args.sven else 
            loss, loss_dict = self.step(batch)
            acc_loss_dict.step(loss_dict)
        return acc_loss_dict.pretty_print(self.args)

    def set_seed_util(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def load_model_util(self, model_name, args):
        """
        Important note:
        This load function will only work for lora models if they are saved in the following pattern:
            <pretrained_base_model_name>-lora<whatever_else>
        """

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
        if model_name in PRETRAINED_MODELS or model_name == 'deepseek':
            model = AutoModelForCausalLM.from_pretrained(model_dir, device_map='auto', trust_remote_code=True)
        else:    
            model = AutoModelForCausalLM.from_pretrained(model_dir, device_map='auto', trust_remote_code=True, **{'vocab_size': len(tokenizer)})
        model.resize_token_embeddings(len(tokenizer))
        return tokenizer, model

    def load_model(self):
        self.tokenizer, self.model = self.load_model_util(self.args.pretrain_name, self.args)
        self.model.train()

        if self.args.kl_loss_weight > 0 and not self.args.sven:
            _, self.ref_model = self.load_model_util(self.args.pretrain_name, self.args)
            self.ref_model.eval()

    def load_dataset(self):
        self.dataset = CodeDataset(self.args, self.tokenizer, 'train')
        self.val_dataset = CodeDataset(self.args, self.tokenizer, 'val')

    def save(self, path):
        """
        For normal models this saves the whole set of weights, for LoRA models it saves the adapter.
        """
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
    def run(self):
        self.load_model()
        self.load_dataset()

        self.args.logger.info(f'Training args {self.args}')

        batch_size = self.args.batch_size
        train_sampler = RandomSampler(self.dataset)
        train_dataloader = DataLoader(self.dataset, sampler=train_sampler, batch_size=batch_size, drop_last=True)

        total_samples = len(self.dataset)
        batch_size = batch_size * self.args.grad_acc_steps
        total_steps = total_samples // batch_size * self.args.num_train_epochs

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if (not any(nd in n for nd in no_decay)) and p.requires_grad],
            'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=total_steps)
        num_params = sum(p.numel() for p in self.model.parameters())
        num_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.args.logger.info('***** Running training *****')
        self.args.logger.info('  Num samples = %d', total_samples)
        self.args.logger.info('  Num epoch = %d', self.args.num_train_epochs)
        self.args.logger.info('  Batch size= 1')
        self.args.logger.info('  Total batch size (w. accumulation) = %d', batch_size)
        self.args.logger.info('  Gradient Accumulation steps = %d', self.args.grad_acc_steps)
        self.args.logger.info('  Total optimization steps = %d', total_steps)
        self.args.logger.info('  Num val samples = %d', len(self.val_dataset))
        self.args.logger.info('  Num parameters = %d', num_params)
        self.args.logger.info('  Num trainable parameters = %d', num_trainable_params)

        global_step, acc_loss_dict = 0, LossDict(self.loss_keys)
        self.set_seed_util(self.args.seed)
        timer = Timer(total_steps)
        timer.start()
        self.model.train()
        for idx in range(self.args.num_train_epochs):
            for step, batch in enumerate(train_dataloader):
                loss, loss_dict = self.step(batch)

                loss /= self.args.grad_acc_steps
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                acc_loss_dict.step(loss_dict)

                if (step+1) % self.args.grad_acc_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()  
                    global_step += 1

                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        acc_loss_pp = acc_loss_dict.pretty_print(self.args)
                        self.args.logger.info('epochs: %s/%d, steps: %s/%d, %s, %s', idx+1, self.args.num_train_epochs, global_step, total_steps, acc_loss_pp, timer)
                        acc_loss_dict.clear()

                    timer.end()
                    timer.start()

            if self.args.save_epochs > 0 and (idx+1) % self.args.save_epochs == 0:
                self.model.eval()
                with torch.no_grad():
                    eval_loss_pp = self.do_eval()
                self.model.train()
                self.args.logger.info('val epoch %s: %s', idx+1, eval_loss_pp)
                output_dir = os.path.join(self.args.output_dir, f'checkpoint-epoch-{idx+1}')
                last_output_dir = os.path.join(self.args.output_dir, f'checkpoint-last')
                self.args.logger.info('Saving model checkpoint to %s and %s', output_dir, last_output_dir)
                self.save(output_dir)
                self.save(last_output_dir)

        if (idx+1) % self.args.save_epochs != 0:
            self.model.eval()
            with torch.no_grad():
                eval_loss_pp = self.do_eval()
            self.args.logger.info('final eval loss: %s', eval_loss_pp)
            # output_dir = os.path.join(self.args.output_dir, f'checkpoint-epoch-{idx+1}')
            last_output_dir = os.path.join(self.args.output_dir, f'checkpoint-last')
            # self.args.logger.info('Saving model checkpoint to %s and %s', output_dir, last_output_dir)
            self.args.logger.info('Saving model checkpoint to %s', last_output_dir)
            # self.save(output_dir)
            self.save(last_output_dir)
