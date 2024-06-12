#!/bin/bash
module load CUDA/12.0.0
module load NCCL/2.16.2-GCCcore-12.2.0-CUDA-12.0.0

echo "DeepSeek Coder 6.7b Instruct"

python generate.py \
    --model deepseek-ai/deepseek-coder-6.7b-instruct \
    --output_path deepseek-coder-6.7b-instruct \
    --dataset infer_datas/gcs_full.jsonl \
    --task 1 2 3 \
    --temperature 0.1 \
    --top_k 50 \
    --top_p 0.2 \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 1 \

echo "DeepSeek Coder 33b Instruct"

python generate.py \
    --model deepseek-ai/deepseek-coder-33b-instruct \
    --output_path deepseek-coder-33b-instruct \
    --dataset infer_datas/gcs_full.jsonl \
    --task 1 2 3 \
    --temperature 0.1 \
    --top_k 50 \
    --top_p 0.2 \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 1 \

echo "CodeLlama 13b Instruct"

python generate.py \
    --model codellama/CodeLlama-13b-Instruct-hf \
    --output_path codellama-13b-instruct \
    --dataset infer_datas/gcs_full.jsonl \
    --task 1 2 3 \
    --temperature 0.1 \
    --top_k 50 \
    --top_p 0.2 \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 1 \

echo "CodeLlama 34b Instruct"

python generate.py \
    --model codellama/CodeLlama-34b-Instruct-hf \
    --output_path codellama-34b-instruct \
    --dataset infer_datas/gcs_full.jsonl \
    --task 1 2 3 \
    --temperature 0.1 \
    --top_k 50 \
    --top_p 0.2 \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 1 \

echo "CodeLlama 70b Instruct"

python generate.py \
    --model codellama/CodeLlama-70b-Instruct-hf \
    --output_path codellama-70b-instruct \
    --dataset infer_datas/gcs_full.jsonl \
    --task 2 3 \
    --temperature 0.1 \
    --top_k 50 \
    --top_p 0.2 \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 1 \

echo "DeepSeek Coder 6.7b Instruct ID SFT Task 2"

python generate.py \
    --model deepseek-ai/deepseek-coder-6.7b-instruct \
    --enable-lora \
    --lora_path deepseek_coder_id_task2 \
    --output_path deepseek-coder-6.7b-instruct-id-sft \
    --dataset infer_datas/gcs_full.jsonl \
    --task 2 \
    --temperature 0.1 \
    --top_k 50 \
    --top_p 0.2 \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 1 \

echo "DeepSeek Coder 6.7b Instruct OOD SFT Task 2"

python generate.py \
    --model deepseek-ai/deepseek-coder-6.7b-instruct \
    --enable-lora \
    --lora_path deepseek_coder_ood_task2 \
    --output_path deepseek-coder-6.7b-instruct-ood-sft \
    --dataset infer_datas/gcs_full.jsonl \
    --task 2 \
    --temperature 0.1 \
    --top_k 50 \
    --top_p 0.2 \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 1 \

echo "DeepSeek Coder 6.7b Instruct ID SFT Task 3"

python generate.py \
    --model deepseek-ai/deepseek-coder-6.7b-instruct \
    --enable-lora \
    --lora_path deepseek_coder_id_task3 \
    --output_path deepseek-coder-6.7b-instruct-id-sft \
    --dataset infer_datas/gcs_full.jsonl \
    --task 3 \
    --temperature 0.1 \
    --top_k 50 \
    --top_p 0.2 \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 1 \

echo "DeepSeek Coder 6.7b Instruct OOD SFT Task 3"

python generate.py \
    --model deepseek-ai/deepseek-coder-6.7b-instruct \
    --enable-lora \
    --lora_path deepseek_coder_ood_task3 \
    --output_path deepseek-coder-6.7b-instruct-ood-sft \
    --dataset infer_datas/gcs_full.jsonl \
    --task 3 \
    --temperature 0.1 \
    --top_k 50 \
    --top_p 0.2 \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 1 \
