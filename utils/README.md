# Quick Usage

Place your results in `/results` directory, and update the `--result_path` in `exec.sh` with your path. Also, modify the log address. 

Then run `bash exec.sh`. Afterward, you can check the run logs in your log file, view the overall results in `eval_total_user.jsonl`, and see the results for each repository in `eval_result_user.jsonl`.


Both JSONL files starting with `eval_result` and `eval_total` contain partial execution results in our paper.

# Files introduction

Lines 38-52 of `exec.py` specify the repository paths, corresponding to the folders in the current directory.

The `/results` folder includes the model-generated outputs we used for testing.

The `/exec_logs` folder saves our the execute log.

The `/benchmark` folder contains testsets ML_bench_full and ML_bench_quarter

The `temp.py` file is not for users, it is used to store the code written by models.

Additionally, the execution process may generate new unnecessary files.
