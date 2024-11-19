import fire
import os
import sys
import json
import re
import asyncio

import pandas as pd
from tqdm import tqdm
from anthropic import AsyncAnthropic
from datasets import load_dataset, Dataset, DatasetDict

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

TASK4_PROMPT = """Suppose you are a data scientist, you will read the following README file and write a script to do the task described by the user.
You want to help me with this task:
{}

In your answer, the script should be generated between ``` and ```. Don't generate multiple code blocks. Most answers can be directly found in the above segment. You might need to modify the code a little bit to make it work.
"""

ANTHROPIC_MODELS = {
    "Anthropic/claude-3-5-sonnet-20241022": "claude-3-5-sonnet-20241022",
    "Anthropic/claude-3-5-haiku-20241022": "claude-3-5-haiku-20241022",
    # "Anthropic/claude-3-opus-20240229": "claude-3-opus-20240229",
}

def read_dialogs_from_dataset(task, cache_dir="cache/ml-bench-merged"):
    dataset_dict = DatasetDict.load_from_disk(cache_dir)
    dataset = dataset_dict['full']  # Assuming the dataset is stored under the 'full' key
    dialogs = pd.DataFrame(dataset)
    
    if task == 1:
        dialogs["prompt"] = dialogs.progress_apply(
            lambda x: [{'role': 'user', 'content': TASK1_PROMPT.format(x['instruction'], x['readme_content'])}], axis=1
        )
    elif task == 2:
        dialogs["prompt"] = dialogs.progress_apply(
            lambda x: [{'role': 'user', 'content': TASK2_PROMPT.format(x['instruction'], x['bm25_result'])}], axis=1
        )
    elif task == 3:
        dialogs["prompt"] = dialogs.progress_apply(
            lambda x: [{'role': 'user', 'content': TASK3_PROMPT.format(x['instruction'], x['oracle'])}], axis=1
        )
    elif task == 4:
        dialogs["prompt"] = dialogs.progress_apply(
            lambda x: [{'role': 'user', 'content': TASK4_PROMPT.format(x['instruction'])}], axis=1
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

async def generate_code_response(client, messages, model, num_repeat=5, max_tokens=64, temperature=1.0, **kwargs):
    async def single_request():
        while True:
            try:
                response = await client.messages.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                )
                return response.content[0].text
            except Exception as e:
                print(f"Error: {e}")
                await asyncio.sleep(1)
                return ""
 

    tasks = [single_request() for _ in range(num_repeat)]
    responses = await asyncio.gather(*tasks)
    return responses

async def main(
    max_tokens = 4096, #The maximum numbers of tokens to generate
    num_repeat: int=5, #The number of samples for each input
    temperature: float=1.0, # [optional] The value used to modulate the next token probabilities.
    cache_dir: str="cache/ml-bench-merged", # The directory where the cached dataset is stored
    **kwargs
):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
    client = AsyncAnthropic(api_key=api_key)

    for task in [1,2,3,4]:
        dialogs = read_dialogs_from_dataset(task, cache_dir)

        async def process_model(model_name, model):
            output_texts = []
            
            for messages in tqdm(dialogs["prompt"], desc=f"Processing {model_name}"):
                output_text = await generate_code_response(
                    client,
                    messages,
                    model,
                    num_repeat=num_repeat,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                )
                output_text = [postprocess_output(output) for output in output_text]
                output_texts.append(output_text)

            dialogs_copy = dialogs.copy()
            dialogs_copy["output"] = output_texts
            result_dialogs = dialogs_copy[["github_id", "id", "output"]]
            with open(f"output/{model_name.replace('/', '_')}-task{task}.jsonl", "w") as f:
                for _, row in result_dialogs.iterrows():
                    f.write(json.dumps(row.to_dict()) + "\n")

        await asyncio.gather(*[process_model(model_name, model) for model_name, model in ANTHROPIC_MODELS.items()])

if __name__ == "__main__":
    fire.Fire(lambda **kwargs: asyncio.run(main(**kwargs)))