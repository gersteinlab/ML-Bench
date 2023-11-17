"""Tools to generate from OpenAI prompts."""

import asyncio
import logging
import os
from typing import Any
import json
import re
import json
import aiolimiter
import openai
import openai.error
from aiohttp import ClientSession
from tqdm.asyncio import tqdm_asyncio
import random
from tqdm import tqdm
import argparse



def get_args_parser():
    parser = argparse.ArgumentParser('gptparsing', add_help=False)
    # Model parameters
    
    parser.add_argument('--readme_type', default='readme', type=str,help='readme or oracle_segment')
    parser.add_argument('--instruction', default='extend_instructions', type=str)
    parser.add_argument('--nturn', default='', type=int,help='n round')
    parser.add_argument('--input_file', default='', type=str,help='input_file')
    parser.add_argument('--answer_file', default='', type=str,help='answer_file')
    parser.add_argument('--parsing_file', default='', type=str,help='parsing_file')
    parser.add_argument('--engine', default='', type=str,help='engine')
    parser.add_argument('--openai_key', default='', type=str,help='key')
    return parser

def perturbation_prompt(question, instruction):
    message = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": question},
    ]
    return message

async def _throttled_openai_chat_completion_acreate(
    model: str,
    messages: list(dict()),
    temperature: float,
    # max_tokens: int,
    top_p: float,
    n: int,
    limiter: aiolimiter.AsyncLimiter,
) -> dict():
    async with limiter:
        for _ in range(10):
            try:
                return await openai.ChatCompletion.acreate(
                    engine=model,
                    messages=messages,
                    temperature=temperature,
                    # max_tokens=max_tokens,
                    n=n,
                    top_p=top_p,
                )
            except openai.error.RateLimitError:
                logging.warning(
                    "OpenAI API rate limit exceeded. Sleeping for 20 seconds."
                )
                sleep_time = random.randint(10, 20)
                await asyncio.sleep(sleep_time)
            except asyncio.exceptions.TimeoutError or openai.error.Timeout or asyncio.TimeoutError:
                logging.warning("OpenAI API timeout. Sleeping for 10 seconds.")
                await asyncio.sleep(10)
            except openai.error.APIError as e:
                logging.warning(f"OpenAI API error: {e}")
                await asyncio.sleep(10)
            except:
                logging.warning("Unknown OpenAI API error. Sleeping for 10 seconds.")
                await asyncio.sleep(10)
        return {"choices": [{"message": {"content": ""}}]}


async def generate_from_openai_chat_completion(
    api_key: str,
    messages,
    engine_name: str,
    temperature: float,
    # max_tokens: int,
    n: int,
    top_p: float,
    requests_per_minute: int = 300,
) -> list():
    """Generate from OpenAI Chat Completion API.

    Args:
        full_contexts: List of full contexts to generate from.
        prompt_template: Prompt template to use.
        model_config: Model configuration.
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """
    openai.api_key = api_key
    session = ClientSession()
    openai.aiosession.set(session)
    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_openai_chat_completion_acreate(
            model=engine_name,
            messages=message,
            temperature=temperature,
            # max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
            n=n
        )
        for message in messages
    ]
    responses = await tqdm_asyncio.gather(*async_responses)
    await session.close()
    
    return_data = []
    for index_i in range(len(responses)):
        cur_data = []
        for index_j in range(n):
            cur_data.append(responses[index_i]["choices"][index_j]["message"]["content"])
        return_data.append(cur_data)
    
    return return_data


def task(str):
    input_str = str
    pattern = r"```python(.*?)```"
    match = re.search(pattern, input_str, re.DOTALL)

    if match:
        result = match.group(1)
        answer = result
    else:
        pattern = r"```bash(.*?)```"
        match = re.search(pattern, input_str, re.DOTALL)
        if match:
            result = match.group(1)
            # print(result)
            answer = result
        else:
            pattern = r"```shell(.*?)```"
            match = re.search(pattern, input_str, re.DOTALL)
            if match:
                result = match.group(1)
                answer = result
            else:
                start_index = input_str.find("```")

                if start_index != -1:
                 
                    result = input_str[start_index + 3:]
                    end_index = result.find("```")
            
                    if end_index != -1:                  
                        final_result = result[:end_index]
                        answer = final_result
                    else:
                        answer = input_str
                else:
                    answer = input_str
                
    return answer

def parsing(args):
    import json
    
    input_json_file = args.answer_file  

    data=[]
    
    with open(input_json_file, 'r') as input_file:
        data = json.load(input_file)

    
    for item in data:
        if 'output' in item and isinstance(item['output'], list):
            item['output'] = [task(output_item) for output_item in item['output']]



    import json


    output_jsonl_file = args.parsing_file  

    with open(output_jsonl_file, 'w') as output_file:
        for item in data:
            output_line = json.dumps(item)
            output_file.write(output_line + '\n')

    print("file save to:", output_jsonl_file)




if __name__ == "__main__":
    messages = [[{"role": "user", "content": f"print the number {i}"}] for i in range(1000)]
    
    data = []
    

    
args = get_args_parser().parse_args()   

openai.api_key = args.openai_key
openai.api_base = "xxxx"
openai.api_type = ""
openai.api_version = ""

template = '''[readme]:{content}
[instruction]:{extend_instructions}

[System]
You are given [readme], you need to carefully see [readme] and choose wirte code or script to implement my [instruction]. 
Please output code or script directly, use markdown to output code without any explanation .'''
final = {}

with open(args.input_file, "r") as fp:
    # data = json.load(fp)
    for line in fp:
        item = json.loads(line)
        data.append(item)

    id = 0
    messages = []
    print("Processing the data.")
    for line in tqdm(data):
        
        
        start_template = template.format(content=line[f'{args.readme_type}'], 
                             extend_instructions=line[f'{args.instruction}']
                             )
        
        
        messages.append([{"role": "user", "content": start_template}])
        


            
    responses = asyncio.run(generate_from_openai_chat_completion(
        api_key=args.openai_key,
        messages=messages, #240
        engine_name=args.engine,
        temperature=1.0,
        top_p=1.0,
        n=args.nturn
    ))

    output_data = []
    
    for index in range(len(responses)):
        output_data.append({
            "github_id": data[index]["github_id"],
            "id": int(data[index]["id"]),
            "output": responses[index],
        })
        
    
    with open(args.answer_file, "w") as fp:
        json.dump(output_data, fp)
        
parsing(args)
