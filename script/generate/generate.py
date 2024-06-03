import argparse
from typing import Any, Dict, List
from huggingface_hub import snapshot_download
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

import re
import json
import pandas as pd
import random
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

tqdm.pandas()

chat_template = {
    "lmsys/vicuna-7b-v1.5": 
"""{% if messages[0]['role'] == 'system' %}
    {% set loop_messages = messages[1:] %}
    {% set system_message = messages[0]['content'].strip() + '\n\n' %}
{% else %}
    {% set loop_messages = messages %}
    {% set system_message = '' %}
{% endif %}

{{ bos_token + system_message }}
{% for message in loop_messages %}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {% endif %}
    
    {% if message['role'] == 'user' %}
        {{ 'USER: ' + message['content'].strip() + '\n' }}
    {% elif message['role'] == 'assistant' %}
        {{ 'ASSISTANT: ' + message['content'].strip() + eos_token + '\n' }}
    {% endif %}
    
    {% if loop.last and message['role'] == 'user' and add_generation_prompt %}
        {{ 'ASSISTANT:' }}
    {% endif %}
{% endfor %}
""",
}


TASK1_PROMPT = """Suppose you are a data scientist, you will read the following README file and write a python script to do the task described by the user.
You want to help me with this task:
{}
Here is the README of this Github Repo, you might refer to it to write the script:
{}

In short format, the code should be generated between ``` and ```. Don't generate multiple code blocks or comments. Please start the code with the following line:
```
"""

TASK2_PROMPT = """Suppose you are a data scientist, you will read the following README file and write a script to do the task described by the user.
You want to help me with this task:
{}
Here is part of a README of this Github Repo, you might refer to it to write the script:
{}

In short format, the script should be generated between ``` and ```. Don't generate multiple code blocks or comments. Please start the code with the following line:
```
"""

TASK3_PROMPT = """Suppose you are a data scientist, you will read the following README file and write a script to do the task described by the user.
You want to help me with this task:
{}
If you want to use the code segment and script in the following segment, you can use it directly:
{}

In short format, the script should be generated between ``` and ```. Don't generate multiple code blocks or comments. Please start the code with the following line:
```
"""

def read_dialogs_from_file(dataset, task):
    dialogs = []
    with open(dataset) as f:
        for line in f:
            dialogs.append(json.loads(line))
    dialogs = pd.DataFrame(dialogs)
    if task == 1:
        dialogs["messages"] = dialogs.progress_apply(
            lambda x: [{'role': 'user', 'content': TASK1_PROMPT.format(x['instruction'], x['readme_content'])}], axis=1
        )
    elif task == 2:
        dialogs["messages"] = dialogs.progress_apply(
            lambda x: [{'role': 'user', 'content': TASK2_PROMPT.format(x['instruction'], x['bm25 result'])}], axis=1
        )
    elif task == 3:
        dialogs["messages"] = dialogs.progress_apply(
            lambda x: [{'role': 'user', 'content': TASK3_PROMPT.format(x['instruction'], x['golden code segment'])}], axis=1
        )
    dialogs = dialogs[["github_id", "id", "messages"]]
    return Dataset.from_pandas(dialogs)

def parse_codeblock(codeblock: str) -> str:
    pattern = re.compile(r"```(?:bash|python|script|shell)?\n(.*?)\n```", re.DOTALL)
    match = pattern.search(codeblock)
    if match:
        return match.group(1)
    return ""


def main(args):
    llm = LLM(
        model=args.model,
        enable_lora=args.enable_lora,
        swap_space=120,
        max_lora_rank=args.max_lora_rank,
        max_num_batched_tokens=65528 if args.enable_lora else None,
        max_model_len=65528 if args.enable_lora else None,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.model in chat_template and tokenizer.chat_template is None:
        tokenizer.chat_template = chat_template.get(args.model, None)
    if args.enable_lora:
        lora_path = args.lora_path or snapshot_download(repo_id=args.repo_id)

    sampling_params = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        seed=args.seed if args.seed is not None else random.randint(0, 1000000),
        max_tokens=args.max_tokens,
        top_k=args.top_k,
        top_p=args.top_p,
        stop=[tokenizer.eos_token]
    )

    def convert_to_prompt(example):
        return {"prompt": tokenizer.apply_chat_template(example["messages"], tokenize=False) + "### Response: "}

    for task in args.task:
        dataset = read_dialogs_from_file(args.dataset, task)
        dataset = dataset.map(convert_to_prompt, num_proc=8)

        outputs = llm.generate(
            dataset["prompt"],
            sampling_params,
            lora_request=LoRARequest("adapter", 1, lora_path) if args.enable_lora else None
        )

        unique = []
        # jsonl
        with open(f"{args.output_path}-task{task}.jsonl", "w") as f:
            for idx, pred in enumerate(outputs):
                output = [parse_codeblock(pred.outputs[i].text) for i in range(len(pred.outputs))]
                f.write(json.dumps({"github_id": dataset["github_id"][idx], "id": dataset["id"][idx], "output": output}) + "\n")
                unique.append(set(output))
        print(f"Task {task} done!")
        print(f"Average unique code blocks: {sum([len(u) for u in unique]) / len(unique)}")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM LoRA adapter example")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf", help="Base model name or path")
    parser.add_argument("--repo-id", type=str, default=None, help="Hugging Face model hub repo ID")
    parser.add_argument('--task', type=int, nargs='+', help='Task numbers')
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA model path")
    parser.add_argument("--output_path", type=str, default="outputs.txt", help="Output path")
    parser.add_argument("--dataset", type=str, default="tatsu-lab/alpaca", help="Dataset name")
    parser.add_argument("--temperature", type=float, default=1, help="Sampling temperature")
    parser.add_argument("--n", type=int, default=5, help="Number of sequences to generate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--best_of", type=int, default=10, help="Best of n")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max tokens to generate")
    parser.add_argument("--enable-lora", action="store_true", help="Enable LoRA")
    parser.add_argument("--max-lora-rank", type=int, default=64, help="Max LoRA rank")
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--pipeline-parallel-size", "-pp", type=int, default=1, help="Pipeline parallel size")
    args = parser.parse_args()

    main(args)
