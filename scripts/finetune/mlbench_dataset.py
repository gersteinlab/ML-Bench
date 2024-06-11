# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import copy
import datasets
import itertools

USER_PROMPT = """
Suppose you are a data scientist, you will read the following README file and write a python script to do the task described by the user.
You want to help me with this task:
{}

Here is the README of this Github Repo, you might refer to it to write the script:
{}
"""
ASSISTANT_ANSWER = """
Here is the script I wrote for you:
```
{}
```
"""
B_INST, E_INST = "[INST]", "[/INST]"

def to_dialog(input, output, readme):
    dialog = []
    dialog.append({"content": USER_PROMPT.format(input, readme), "role": "user"})
    dialog.append({"content": ASSISTANT_ANSWER.format(output), "role": "assistant"})
    return {"dialog": dialog}

def tokenize_dialog(dialog, tokenizer):
    prompt_tokens = [tokenizer.encode(f"{tokenizer.bos_token}{B_INST} {(prompt['content']).strip()} {E_INST}", add_special_tokens=False) for prompt in dialog[::2]]
    answer_tokens = [tokenizer.encode(f"{answer['content'].strip()} {tokenizer.eos_token}", add_special_tokens=False) for answer in dialog[1::2]]
    dialog_tokens = list(itertools.chain.from_iterable(zip(prompt_tokens, answer_tokens)))
    # Add labels, convert prompt token to -100 in order to ignore in loss function
    labels_tokens = [len(c)*[-100,] if i % 2 == 0 else c for i,c in enumerate(dialog_tokens)]

    combined_tokens = {
        "input_ids": list(itertools.chain(*(t for t in dialog_tokens))),
        "labels": list(itertools.chain(*(t for t in labels_tokens))),
    }

    return dict(combined_tokens, attention_mask=[1]*len(combined_tokens["input_ids"]))


def get_custom_dataset(dataset_config, tokenizer, split):
    dataset = datasets.load_from_disk(dataset_config.data_path)
    
    if dataset_config.task == 1:
        dataset = dataset.map(lambda sample: {
            "readme": sample["readme"],
            "output": sample["output"],
            'input': sample['instructions'],
            },
            batched=True,
            remove_columns=list(dataset.features),)
    elif dataset_config.task == 2:
        dataset = dataset.map(lambda sample: {
            "readme": sample["bm25_result"],
            "output": sample["output"],
            'input': sample['instructions'],
            },
            batched=True,
            remove_columns=list(dataset.features),)
    elif dataset_config.task == 3:
        dataset = dataset.map(lambda sample: {
            "readme": sample["oracle_segment"],
            "output": sample["output"],
            'input': sample['instructions'],
            },
            batched=True,
            remove_columns=list(dataset.features),)

    dataset = dataset.map(lambda x: to_dialog(x["input"], x["output"], x["readme"]), remove_columns=list(dataset.features))
    dataset = dataset.map(lambda x: tokenize_dialog(x["dialog"], tokenizer), remove_columns=list(dataset.features))
    split_dataset = dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)
    return split_dataset[split]