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
