import os
import sys
import argparse
from transformers.utils.import_utils import subprocess
import yaml
import torch
from io import StringIO
import subprocess
from tqdm import tqdm

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_checkpoint', type=str, default='bigcode/starcoderbase-1b')
  parser.add_argument('--eval_type', type=str, default='mbpp', help='mbpp/human_eval')
  parser.add_argument('--num_pass', type=int, default=1)
  parser.add_argument('--temp', type=float, default=0.2)
  args = parser.parse_args()
  return args

def test_program(program_text):
  with open('temp_program.py', 'w') as f:
    f.write(program_text)
  try:
    # Run the Python file using subprocess and capture output
    result = subprocess.run(['python', 'temp_program.py'],  # Specify the Python interpreter and the file path
      text=True,              # Capture output as a string
      capture_output=True,    # Capture stdout and stderr
      check=True              # Raise an error if the process exits with a non-zero status
    )
    return True
  except Exception:
    return False

if __name__ == '__main__':
  from transformers import AutoModelForCausalLM, AutoTokenizer
  args = parse_args()

  device = "cuda" # for GPU usage or "cpu" for CPU usage
  tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
  model = AutoModelForCausalLM.from_pretrained(args.model_checkpoint).to(device)
  count = 0
  total = 0
  eval_type = args.eval_type


  dataset_dir = f'data_eval/{eval_type}/'

  for eval_file in tqdm(os.listdir(dataset_dir)):
    with open(os.path.join(dataset_dir, eval_file)) as file:
      data = yaml.safe_load(file)
    if data['language'] != 'py':
      continue
    total += 1
    prompt = data['prompt']
    test = data['tests']
    stop_tokens = data['stop_tokens']
    input_tokens = tokenizer.encode(prompt.strip(), return_tensors='pt').to(device)
    outputs = model.generate(input_tokens, do_sample=True,
              num_return_sequences=args.num_pass,
              temperature=args.temp,
              max_new_tokens=256,
              top_p=0.95,
              pad_token_id=tokenizer.eos_token_id,
              eos_token_id=tokenizer.eos_token_id,
              use_cache=True,
          )
    outputs[:, :input_tokens.shape[1]] = input_tokens
    results = outputs[:, input_tokens.shape[1]:]
    for i in range(len(results)):
      result = results[i]
      if tokenizer.eos_token_id in results[i]:
        idx = torch.where(result == tokenizer.eos_token_id)[0][0].item()
        result = result[:idx]
      program_text = tokenizer.decode(result, skip_special_tokens=True)
      for stop_token in stop_tokens:
        if stop_token in program_text:
          program_text = program_text[:program_text.find(stop_token)]
      program_text = prompt + program_text + '\n' + test
      if test_program(program_text):
        count += 1
        break
        
  print('Eval:', eval_type)
  print('Total:', total)
  print('Count:', count)
  print('Acc:', count / total)
