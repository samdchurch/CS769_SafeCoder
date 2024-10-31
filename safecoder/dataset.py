import os
import json
import torch
import difflib
from torch.utils.data import Dataset
import numpy as np
from random import shuffle

from safecoder.constants import CWES_TRAINED, CWES_NEW_TRAINED, FUNC, GOOD, BAD, PROMPT_INPUT, PROMPT_NO_INPUT, SAFE_DESC_DATASETS
from safecoder.utils import visualize_pair, visualize_weights, inspect_cwe_dist
import pickle

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
