# CS769_SafeCoder

## Setup (from SafeCoder)
First, install Python dependencies:
```console
pip install -r requirements.txt
pip install -e .
```
Then, install [GitHub CodeQL](https://codeql.github.com/), which will be used for evaluating the security of LLM-generated code:
```console
./setup_codeql.sh
```
Finally, set up different programming languages studied in this work (`sudo` rights required):
```console
./setup_langs.sh
```

## Training
### Full fine-tuning
```console
python train.py --pretrain_name starcoderbase-1b --output_name starcoderbase-1b-safecoder --datasets evol sec-desc sec-new-desc
```
### LoRA
```console
python train_lora.py --pretrain_name starcoderbase-1b --lora --r 16 --output_name starcoderbase-1b-safecoder-lora-16 --datasets evol sec-desc sec-new-desc
```
### Prefix Tuning
```console
python train_prefix_tuning.py --pretrain_name starcoderbase-1b --prefix_tuning --num_prefix_tokens 16 --output_name starcoderbase-1b-safecoder-prefix-16 --datasets evol sec-desc sec-new-desc
```
### DPO
