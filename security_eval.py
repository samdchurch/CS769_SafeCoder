import os
import json
import tempfile
import subprocess

extension_languages = {'js': 'javascript', 'py': 'python', 'c': 'cpp', 'cpp': 'cpp'}

def run_query(database_dir, ql_file, output_file):
  command = ['codeql/codeql', 'query', 'run', f'--database={database_dir}', f'--output={output_file}', ql_file]
  subprocess.run(command, stdout=subprocess.DEVNULL)
  #print(f"Query ran successfully. Results saved to {output_file}")

def bqrs_to_json(bqrs_file, json_file):
  command = ["codeql/codeql", "bqrs", "decode", "--format=json", "--quiet", "--output", json_file, bqrs_file]
  subprocess.run(command, stdout=subprocess.DEVNULL)

  #print(f"Results decoded to JSON format at {json_file}")

def evaluate_result(program_text, info):
  extension = info['language']
  if extension != 'py':
    return None
  temp_code_path = create_temp_code_file(program_text, extension)
  database_dir = tempfile.mkdtemp()
  create_database(temp_code_path, database_dir, extension_languages[extension])
  output_bqrs = tempfile.NamedTemporaryFile(delete=False, suffix=".bqrs").name
  output_json = tempfile.NamedTemporaryFile(delete=False, suffix=".json").name
  check_ql_file = '/'.join(info['check_ql'].split('/')[1:])
  run_query(database_dir, check_ql_file, output_bqrs)
  bqrs_to_json(output_bqrs, output_json)

  with open(output_json) as f:
    results = json.load(f)
  print(results)
  for key in results.keys():
    if 'tuples' in results[key]:
      if len(results[key]['tuples']) != 0:
        return 0
  return 1


def create_database(temp_code_file, database_dir, language):
  command = ['codeql/codeql', 'database', 'create', database_dir, f'--language={language}', '--quiet', '--overwrite',
            '--source-root', os.path.dirname(temp_code_file)]
  subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
  print(f"CodeQL database created at {database_dir}")

def create_temp_code_file(code_str, extension):
  temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{extension}')
  temp_file.write(code_str.encode('utf-8'))
  temp_file.close()
  return temp_file.name

def evaluate_example(model, tokenizer, example_path, device):
  extension = example_path.split('-')[-1]
  with open(f'{example_path}/file_context.{extension}') as f:
    file_context = f.read()
  with open(f'{example_path}/func_context.{extension}') as f:
    func_context = f.read()
  with open(f'{example_path}/info.json') as f:
    info = json.load(f)

  input_text = file_context + func_context
  input_tokens = tokenizer.encode(input_text, return_tensors='pt').to(device)
  print(input_tokens.shape)
  outputs = model.generate(input_tokens, do_sample=True,
                num_return_sequences=20,
                temperature=0.4,
                max_new_tokens=256,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

  outputs[:, :input_tokens.shape[1]] = input_tokens

  # store result in temporary text file
  program_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
  return evaluate_result(program_text, info)


def evaluate_model(model, tokenizer, sec_eval_folder, device):
  scores = []
  for training_type_folder in os.listdir(sec_eval_folder):
    training_type_path = os.path.join(sec_eval_folder, training_type_folder)
    for security_type_folder in os.listdir(training_type_path):
      security_type_path = os.path.join(training_type_path, security_type_folder)
      for example_folder in os.listdir(security_type_path):
        example_path = os.path.join(security_type_path, example_folder)
        result = evaluate_example(model, tokenizer, example_path, device)
        if result is not None:
          scores.append(result)
          with open('sec_eval_results.txt', 'w') as f:
            f.write(str(scores))

if __name__ == '__main__':
  from transformers import AutoModelForCausalLM, AutoTokenizer

  checkpoint = "bigcode/starcoderbase-1b"
  device = "cuda" # for GPU usage or "cpu" for CPU usage

  tokenizer = AutoTokenizer.from_pretrained(checkpoint)
  model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

  sec_eval_folder = 'data_eval/sec_eval'
  evaluate_model(model, tokenizer, sec_eval_folder, device)
  
        