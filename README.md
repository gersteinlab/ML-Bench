# ML-Bench: Large Language Models Leverage Open-source Libraries for Machine Learning Tasks





## GPT calling

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


## tools

### get bm25 result

Run `python script/tools/bm25.py` to generate BM25 results for the instructions and readme. Ensure to update the dataset path and include the BM25 result dataset.

### crawl readme files from github repository

Run `python script/tools/crawl.py` to fetch readme files from a specific GitHub repository. You'll need to modify the `url` within the code to retrieve the desired readme files.
