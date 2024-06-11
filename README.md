# ML-Bench: Large Language Models Leverage Open-source Libraries for Machine Learning Tasks

<p align="center">
   📖 <a href="https://arxiv.org/abs/2311.09835" target="_blank">Paper</a>  • 🚀 <a href="https://ml-bench.github.io/" target="_blank">Github Page</a>  • 📊 <a href="https://huggingface.co/datasets/super-dainiu/ml-bench" target="_blank">Data</a> 
</p>

![Alt text](https://github.com/gersteinlab/ML-Bench/blob/master/assets/image.png)

## OpenAI Calling

You can use the following script to reproduce OpenAI's performance on this task：
```python
bash script/openai/run.sh
```

You need to change parameter settings in `script/openai/run.sh` :

* `type`: Choose from quarter or full.

* `model`: Model name 

* `input_file`: File path of dataset

* `answer_file`: Original answer json format from GPT.

* `parsing_file`: Post-process the output of GPT in jsonl format to obtain executable code segments.

* `readme_type`: Choose from oracle_segment and readme

  *# `oracle_segment`: The code paragraph in the readme that is most relevant to the task*

  *# `readme`: The entire text of the readme in the repository where the task is located*

* `engine_name`: Choose from gpt-35-turbo-16k and gpt-4-32.

* `n_turn`: GPT returns the number of executable codes (5 times in the paper experiment).

* `openai_key`: Your key.

## Open Source Model Fine-tuning
Please refer to [finetune](script/finetune/README.md) for details.

## Tools

### Get BM25 result

Run `python utils/bm25.py` to generate BM25 results for the instructions and readme. Ensure to update the original dataset `path` and output `path` which includes the BM25 results.

### Crawl README files from github repository

Run `python utils/crawl.py` to fetch readme files from a specific GitHub repository. You'll need to modify the `url` within the code to retrieve the desired readme files.

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


