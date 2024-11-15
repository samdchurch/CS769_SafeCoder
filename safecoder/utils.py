import os
import ast
import sys
import torch
import random
import difflib
import logging
import tempfile
import warnings
import sacrebleu
import subprocess
import numpy as np
from tabulate import tabulate
from termcolor import colored
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from .constants import PRETRAINED_MODELS, CHAT_MODELS

logger = logging.getLogger()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def set_logging(args, log_file):
    handlers = []
    handlers.append(logging.StreamHandler(stream=sys.stdout))
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=handlers
    )
    args.logger = logger