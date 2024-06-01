# ML-Bench: Large Language Models Leverage Open-source Libraries for Machine Learning Tasks

<p align="center">
   📖 <a href="https://arxiv.org/abs/2311.09835" target="_blank">Paper</a>  • 🚀 <a href="https://ml-bench.github.io/" target="_blank">Github Page</a>  • 📊 <a href="https://drive.google.com/drive/folders/1e86FhLjxXK837SgR8a29cztx9UfxPQzS" target="_blank">Data</a> 
</p>

![Alt text](https://github.com/gersteinlab/ML-Bench/blob/master/assets/image.png)

## Execution Env and data

We have annotated the ML-Bench with new data, filtered and modified it, and we will subsequently update it with the new execution environment and data. 

The execution environment in old version (same version as arxiv paper 2311) can be found in ./Execution env, for data in old version please refer to https://drive.google.com/drive/folders/1e86FhLjxXK837SgR8a29cztx9UfxPQzS?usp=drive_link .

## GPT Calling

You can use the following script to reproduce GPT's performance on this task：
```python
sh script/GPT/run.sh
```

You need to change parameter settings in `script/GPT/run.sh` :

* type: Choose from quarter or full.

* model: Model name 

* input_file: File path of dataset

* answer_file: Original answer json format from GPT.

* parsing_file: Post-process the output of GPT in jsonl format to obtain executable code segments.

* readme_type: Choose from oracle_segment and readme

  *# oracle_segment: The code paragraph in the readme that is most relevant to the task*

  *# readme: The entire text of the readme in the repository where the task is located*

* engine_name: Choose from gpt-35-turbo-16k and gpt-4-32.

* n_turn: GPT returns the number of executable codes (5 times in the paper experiment).

* openai_key: Your key.

## CodeLlama-7b Fine-tuning
Please refer to [CodeLlama-7b](script/codellama/README.md) for details.

## Tools

### Get BM25 result

Run `python script/tools/bm25.py` to generate BM25 results for the instructions and readme. Ensure to update the original dataset `path` and output `path` which includes the BM25 results.

### Crawl README files from github repository

Run `python script/tools/crawl.py` to fetch readme files from a specific GitHub repository. You'll need to modify the `url` within the code to retrieve the desired readme files.

## Cite Us
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


