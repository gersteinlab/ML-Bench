# OpenAI Calling

To reproduce OpenAI's performance on this task, use the following script:
```bash
bash script/openai/run.sh
```

## Parameter Settings

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