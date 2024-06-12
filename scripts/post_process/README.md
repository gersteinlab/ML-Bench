# Post-Processing for ML-LLM-Bench Dataset

This folder contains the post-processing scripts for the ML-LLM-Bench dataset. You can load the dataset using the following code:

```python
from datasets import load_dataset

ml_bench = load_dataset("super-dainiu/ml-bench")    # splits: ['full', 'quarter']
```

The dataset contains the following columns:
- `github_id`: The ID of the GitHub repository.
- `github`: The URL of the GitHub repository.
- `repo_id`: The ID of the sample within each repository.
- `id`: The unique ID of the sample in the entire dataset.
- `path`: The path to the corresponding folder in LLM-Bench.
- `arguments`: The arguments specified in the user requirements.
- `instruction`: The user instructions for the task.
- `oracle`: The oracle contents relevant to the task.
- `type`: The expected output type based on the oracle contents.
- `output`: The ground truth output generated based on the oracle contents.
- `prefix_code`: The code snippet for preparing the execution environment

If you want to run ML-LLM-Bench, you need to do post-processing on the dataset. You can use the following code to post-process the dataset:

```bash
bash scripts/post_process/prepare.sh
```

