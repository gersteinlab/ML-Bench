# Open Source Model Fine-tuning

## Prerequisites
Llama-recipes provides a pip distribution for easy install and usage in other projects. Alternatively, it can be installed from source.

### Install with pip
```
pip install --extra-index-url https://download.pytorch.org/whl/test/cu118 llama-recipes
```
### Install from source
To install from source e.g. for development use this command. We're using hatchling as our build backend which requires an up-to-date pip as well as setuptools package.
```
git clone https://github.com/facebookresearch/llama-recipes
cd llama-recipes
pip install -U pip setuptools
pip install --extra-index-url https://download.pytorch.org/whl/test/cu118 -e .
```

## Fine-tuning
By definition, we have three tasks in the paper.
* Task 1: Given a task description + Code, generate a code snippet.
* Task 2: Given a task description + Retrieval, generate a code snippet.
* Task 3: Given a task description + Oracle, generate a code snippet.

You can use the following script to reproduce CodeLlama-7b's fine-tuning performance on this task：
```bash
torchrun --nproc_per_node 2 finetuning.py \
    --use_peft \
    --peft_method lora \
    --enable_fsdp \
    --model_name codellama/CodeLlama-7b-Instruct-hf \
    --context_length 8192 \
    --dataset mlbench_dataset \
    --output_dir OUTPUT_PATH \
    --task TASK \
    --data_path DATA_PATH \
```

You need to change parameter settings of `OUTPUT_PATH`, `TASK` and `DATA_PATH` correspondingly.
* `OUTPUT_DIR`: The directory to save the model.
* `TASK`: Choose from `1`, `2` and `3`.
* `DATA_PATH`: The directory of the dataset.

## Inference
You can use the following script to reproduce CodeLlama-7b's inference performance on this task：
```bash
python chat_completion.py \
    --model_name 'codellama/CodeLlama-7b-Instruct-hf' \
    --peft_model PEFT_MODEL \
    --prompt_file PROMPT_FILE \
    --task TASK \
```

You need to change parameter settings of `PEFT_MODEL`, `PROMPT_FILE` and `TASK` correspondingly.
* `PEFT_MODEL`: The path of the PEFT model.
* `PROMPT_FILE`: The path of the prompt file.
* `TASK`: Choose from `1`, `2` and `3`.

You can also refer to `vllm/generate.sh` to optimize the throughput and memory usage of the model.