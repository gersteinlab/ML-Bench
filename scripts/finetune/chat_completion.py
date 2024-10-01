# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire
import os
import sys
import json
import re

import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from llama_recipes.inference.chat_utils import format_tokens
from llama_recipes.inference.model_utils import load_model, load_peft_model

tqdm.pandas()

TASK1_PROMPT = """Suppose you are a data scientist, you will read the following README file and write a python script to do the task described by the user.
You want to help me with this task:
{}
Here is the README of this Github Repo, you might refer to it to write the script:
{}

In your answer, the code should be generated between ``` and ```. Don't generate multiple code blocks
"""

TASK2_PROMPT = """Suppose you are a data scientist, you will read the following README file and write a script to do the task described by the user.
You want to help me with this task:
{}
Here is part of a README of this Github Repo, you might refer to it to write the script:
{}

In your answer, the script should be generated between ``` and ```. Don't generate multiple code blocks
"""

TASK3_PROMPT = """Suppose you are a data scientist, you will read the following README file and write a script to do the task described by the user.
You want to help me with this task:
{}
If you want to use the code segment and script in the following segment, you can use it directly:
{}

In your answer, the script should be generated between ``` and ```. Don't generate multiple code blocks. Most answers can be directly found in the above segment. You might need to modify the code a little bit to make it work.
"""

def read_dialogs_from_file(prompt_file, task):
    dialogs = []
    with open(prompt_file) as f:
        for line in f:
            dialogs.append(json.loads(line))
    dialogs = pd.DataFrame(dialogs)
    if task == 1:
        dialogs["prompt"] = dialogs.progress_apply(
            lambda x: [{'role': 'user', 'content': TASK1_PROMPT.format(x['instructions'], x['readme_content'])}], axis=1
        )
    elif task == 2:
        dialogs["prompt"] = dialogs.progress_apply(
            lambda x: [{'role': 'user', 'content': TASK2_PROMPT.format(x['extend_instructions'], x['bm25_result'])}], axis=1
        )
    elif task == 3:
        dialogs["prompt"] = dialogs.progress_apply(
            lambda x: [{'role': 'user', 'content': TASK3_PROMPT.format(x['instructions'], x['oracle_segment'])}], axis=1
        )
    dialogs = dialogs[["github_id", "id", "prompt"]]
    return dialogs

def postprocess_output(output):
    # find the code between ``` and ```
    code = re.findall(r"```python\n*(.*?)\n*```", output, re.DOTALL) or re.findall(r"```bash\n*(.*?)\n*```", output, re.DOTALL) or re.findall(r"```script\n*(.*?)\n*```", output, re.DOTALL) or re.findall(r"```\n*(.*?)\n*```", output, re.DOTALL)
    if code:
        code = code[0]
    else:
        code = ''
    return code

def generate_code_response(input_ids, model, tokenizer, num_repeat=5, max_len=4096, max_new_tokens=64, do_sample=True, top_p=1.0, temperature=1.0, use_cache=True, top_k=50, repetition_penalty=1.0, length_penalty=3, **kwargs):
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_ids = input_ids.unsqueeze(0)
    if input_ids.shape[-1] > max_len:
        input_ids = torch.cat([input_ids[:, :max_len - max_new_tokens], input_ids[:, -30:]], dim=-1)
    input_ids = input_ids.to(model.device)
    outputs = []
    with torch.no_grad():
        for _ in range(num_repeat):
            output = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    top_p=top_p,
                    temperature=temperature,
                    use_cache=use_cache,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    pad_token_id=tokenizer.eos_token_id,
                    **kwargs
            )
            outputs.append(tokenizer.decode(output[0, input_ids.shape[-1]:], skip_special_tokens=True))
    return outputs

def main(
    model_name,
    task: int,  # 1, 2, 3, 4
    peft_model: str=None,
    quantization: bool=False,
    max_new_tokens = 1024, #The maximum numbers of tokens to generate
    min_new_tokens:int=0, #The minimum numbers of tokens to generate
    num_repeat: int=5, #The number of samples for each input
    prompt_file: str=None,
    seed: int=42, #seed value for reproducibility
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=1.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=1.0, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation.
    enable_fsdp: bool=False, # Enable Fully Sharded Data Parallelism
    use_fast_kernels: bool = False, # Enable using SDPA from PyTorch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    **kwargs
):
    if prompt_file is not None:
        assert os.path.exists(
            prompt_file
        ), f"Provided Prompt file does not exist {prompt_file}"

        dialogs = read_dialogs_from_file(prompt_file, task)

    # Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    model = load_model(model_name, quantization)
    if peft_model:
        model = load_peft_model(model, peft_model)

    if use_fast_kernels:
        """
        Setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels 
        based on the hardware being used. This would speed up inference when used for batched inputs.
        """
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model)   
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dialogs["input_ids"] = format_tokens(
        dialogs["prompt"], tokenizer
    )
    
    # sort by length of input_ids
    dialogs = dialogs.sort_values(by=["input_ids"], key=lambda x: x.str.len(), ascending=False)

    output_texts = []
    
    for tokens in tqdm(dialogs["input_ids"]):
        output_text = generate_code_response(
            tokens,
            model,
            tokenizer,
            num_repeat=num_repeat,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            use_cache=use_cache,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            **kwargs
        )
        output_text = [postprocess_output(output) for output in output_text]
        output_texts.append(output_text)
        print("\n==================================\n")
        print("Generated code:")
        for i, output in enumerate(output_text):
            print('\t', i + 1, output)

    dialogs["output"] = output_texts
    dialogs = dialogs[["github_id", "id", "output"]]
    with open(f"output/{peft_model}-task{task}.jsonl", "w") as f:
        for _, row in dialogs.iterrows():
            f.write(json.dumps(row.to_dict()) + "\n")


if __name__ == "__main__":
    fire.Fire(main)
    