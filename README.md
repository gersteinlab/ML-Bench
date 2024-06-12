# ML-Bench: Large Language Models Leverage Open-source Libraries for Machine Learning Tasks

<p align="center">
   📖 <a href="https://arxiv.org/abs/2311.09835" target="_blank">Paper</a>  • 🚀 <a href="https://ml-bench.github.io/" target="_blank">Github Page</a>  • 📊 <a href="https://huggingface.co/datasets/super-dainiu/ml-bench" target="_blank">Data</a> 
</p>

![Alt text](https://github.com/gersteinlab/ML-Bench/blob/master/assets/distribution.png)

## Table of Contents
- 📋 [Prerequisites](#📋-prerequisites)
- 🦙 [ML-LLM-Bench](#🦙-ml-llm-bench)
  - 🌍 [Environment Setup](#🌍-environment-setup)
  - 📞 [API Calling](#📞-api-calling)
  - 🔧 [Open Source Model Fine-tuning](#🔧-open-source-model-fine-tuning)
    - 📋 [Prerequisites](#📋-prerequisites-1)
    - 🏋️ [Fine-tuning](#🏋️-fine-tuning)
    - 🔍 [Inference](#🔍-inference)
- 🤖 [ML-Agent-Bench](#🤖-ml-agent-bench)
  - 🌍 [Environment Setup](#🌍-environment-setup-1)
- 🛠️ [Utils for Data Curations](#🛠️-utils-for-data-curations)
- 📝 [Cite Us](#📝-cite-us)
- 📜 [License](#📜-license)


## 📋 Prerequisites

  To clone this repository with all its submodules, use the `--recurse-submodules` flag:

  ```bash
  git clone --recurse-submodules https://github.com/gersteinlab/ML-Bench.git
  cd ML-Bench
  ```

  If you have already cloned the repository without the `--recurse-submodules` flag, you can run the following commands to fetch the submodules:

  ```bash
  git submodule update --init --recursive
  ```

## 🦙 ML-LLM-Bench

### 📋 Prerequisites

   After clone submodules, you can run 

   `cd utils`

   `bash crawl_raw_repo_and_merge_benchmark.sh` to generate full and quarter benchmark into `merged_full_benchmark.jsonl` and `merged_quarter_benchmark.jsonl`

### 🌍 Environment Setup


   To run the ML-LLM-Bench Docker container, you can use the following command:
   
   ```bash
   docker pull public.ecr.aws/i5g0m1f6/ml-bench
   docker run -it -v ML_Bench:/deep_data public.ecr.aws/i5g0m1f6/ml-bench /bin/bash
   ```

   
   


### 📞 API Calling

To reproduce OpenAI's performance on this task, use the following script:
```bash
bash script/openai/run.sh
```

You need to change the parameter settings in `script/openai/run.sh`:

- `type`: Choose from `quarter` or `full`.
- `model`: Model name.
- `input_file`: File path of the dataset.
- `answer_file`: Original answer in JSON format from GPT.
- `parsing_file`: Post-process the output of GPT in JSONL format to obtain executable code segments.
- `readme_type`: Choose from `oracle_segment` and `readme`.
  - `oracle_segment`: The code paragraph in the README that is most relevant to the task.
  - `readme`: The entire text of the README in the repository where the task is located.
- `engine_name`: Choose from `gpt-35-turbo-16k` and `gpt-4-32`.
- `n_turn`: Number of executable codes GPT returns (5 times in the paper experiment).
- `openai_key`: Your OpenAI API key.

Please refer to [openai](scripts/openai/README.md) for details.

### 🔧 Open Source Model Fine-tuning

#### 📋 Prerequisites
Llama-recipes provides a pip distribution for easy installation and usage in other projects. Alternatively, it can be installed from the source.

1. **Install with pip**
```
pip install --extra-index-url https://download.pytorch.org/whl/test/cu118 llama-recipes
```
2. **Install from source**
To install from source e.g. for development use this command. We're using hatchling as our build backend which requires an up-to-date pip as well as setuptools package.
```
git clone https://github.com/facebookresearch/llama-recipes
cd llama-recipes
pip install -U pip setuptools
pip install --extra-index-url https://download.pytorch.org/whl/test/cu118 -e .
```

#### 🏋️ Fine-tuning
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

You need to change the parameter settings of `OUTPUT_PATH`, `TASK`, and `DATA_PATH` correspondingly.
* `OUTPUT_DIR`: The directory to save the model.
* `TASK`: Choose from `1`, `2` and `3`.
* `DATA_PATH`: The directory of the dataset.

#### 🔍 Inference
You can use the following script to reproduce CodeLlama-7b's inference performance on this task：
```bash
python chat_completion.py \
    --model_name 'codellama/CodeLlama-7b-Instruct-hf' \
    --peft_model PEFT_MODEL \
    --prompt_file PROMPT_FILE \
    --task TASK \
```

You need to change the parameter settings of `PEFT_MODEL`, `PROMPT_FILE`, and `TASK` correspondingly.
* `PEFT_MODEL`: The path of the PEFT model.
* `PROMPT_FILE`: The path of the prompt file.
* `TASK`: Choose from `1`, `2` and `3`.

Please refer to [finetune](scripts/finetune/README.md) for details.

## 🤖 ML-Agent-Bench
### 🌍 Environment Setup

To run the ML-Agent-Bench Docker container, you can use the following command:

```bash
docker pull public.ecr.aws/i5g0m1f6/ml-bench
docker run -it public.ecr.aws/i5g0m1f6/ml-bench /bin/bash
```

This will pull the latest ML-Agent-Bench Docker image and run it in an interactive shell. The container includes all the necessary dependencies to run the ML-Agent-Bench codebase.

For ML-Agent-Bench in OpenDevin, please refer to the [OpenDevin setup guide](https://github.com/OpenDevin/OpenDevin/blob/main/evaluation/ml_bench/README.md).

Please refer to [envs](envs/README.md) for details.


## 🛠️ Utils for Data Curations

1. **Get BM25 result**

  Run `python utils/bm25.py` to generate BM25 results for the instructions and readme. Ensure to update the original dataset `path` and output `path` which includes the BM25 results.

2. **Crawl README files from github repository**

  Run `python utils/crawl.py` to fetch readme files from a specific GitHub repository. You'll need to modify the `url` within the code to retrieve the desired readme files.

3. **Crawl raw repositories**

  Run `bash utils/crawl_raw_repo.sh` to clone repositories and write repositories to txts.

4. **Calculate number of tokens**

  Run `bash utils/calculate_num_tokens.sh` to calculate the number of tokens statistics in the dataset.

## 📝 Cite Us
This project is inspired by some related projects. We would like to thank the authors for their contributions. If you find this project or dataset useful, please cite it:

```
@article{liu2023mlbench,
      title={ML-Bench: Evaluating Large Language Models for Code Generation in Repository-Level Machine Learning Tasks}, 
      author={Yuliang Liu and Xiangru Tang and Zefan Cai and Junjie Lu and Yichi Zhang and Yanjun Shao and Zexuan Deng and Helan Hu and Zengxian Yang and Kaikai An and Ruijun Huang and Shuzheng Si and Sheng Chen and Haozhe Zhao and Zhengliang Li and Liang Chen and Yiming Zong and Yan Wang and Tianyu Liu and Zhiwei Jiang and Baobao Chang and Yujia Qin and Wangchunshu Zhou and Yilun Zhao and Arman Cohan and Mark Gerstein},
      year={2023},
      journal={arXiv preprint arXiv:2311.09835},
}
```

## 📜 License

Distributed under the MIT License. See [`LICENSE`](./LICENSE) for more information.


