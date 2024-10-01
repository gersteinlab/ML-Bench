import json
import subprocess
import logging
import psutil
import argparse
from datetime import datetime
import time
import numpy as np
import pdb
from tqdm import tqdm
import os
from datasets import DatasetDict
from typing import List, Dict, Tuple, Any, Optional

# Constants and configurations
EXEC_TIME = 10
LIST_LEN = 5
ID2CONDA: Dict[int, str] = {
    1: "dgl_DS", 2: "bert_DS", 3: "lavis_DS", 4: "if_DS", 5: "V2V_DS",
    6: "esm_DS", 7: "OP_DS", 8: "TSL_DS", 9: "EAP_DS", 10: "PG_DS",
    11: "PIM_DS", 13: "L3_DS", 14: "MZ2_DS", 15: "GSA2_DS"
}
ID2PATH: Dict[int, str] = {
    1: "repos/dgl", 2: "repos/bert", 3: "repos/lavis", 4: "repos/if",
    5: "repos/vid2vid", 6: "repos/esm", 7: "repos/open_clip",
    8: "repos/Time-Series-Library", 9: "repos/External-Attention-pytorch",
    10: "repos/PyTorch-GAN", 11: "repos/pytorch-image-models",
    13: "repos/learning3d", 14: "repos/muzic",
    15: "repos/Grounded-Segment-Anything"
}

logging.basicConfig(filename='exec.log', level=logging.INFO)


def terminate_process_tree(pid: int) -> None:
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        return
    for child in children:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            continue
    gone, still_alive = psutil.wait_procs(children, timeout=5)
    for process in still_alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            continue

def filter_valid_data(ground_truth_data: List[Dict[str, Any]], error_path: str) -> List[Dict[str, Any]]:
    valid_data = []
    invalid_data = []
    error_count = 0
    for data in ground_truth_data:
        if 'path' not in data:
            pdb.set_trace()
        if len(data["output"]) == LIST_LEN:
            valid_data.append(data)
        else:
            error_count += 1
            logging.info(f"{data} is error and it will be delete. Error json data will be written in error_data.jsonl.")
            print(f"{data} is error and it will be delete. Error json data will be written in error_data.jsonl.")
            invalid_data.append(data)
    
    with open(error_path, "a", encoding="utf-8") as file:
        file.write(f"Error data counts is {error_count}\n")
        logging.info(f"Error data counts is {error_count}\n")
        json.dump(invalid_data, file)
        file.write("\n")
    return valid_data

def run_python_code(repo_output: str, id: int, github_id: int, path: str, conda_env: str) -> bool:
    with open(path + "/temp.py", "w", encoding="utf-8") as file:
        file.write(repo_output)
    
    try:
        cmd = f"cd {path}; source activate {conda_env}; python temp.py"
        sub = subprocess.Popen(["bash", "-c", cmd], stdout=subprocess.PIPE)
        
        timeout = 4 if github_id == 5 else 7 if github_id == 11 else EXEC_TIME
        returncode = sub.wait(timeout=timeout)
        
        if returncode == 0:
            logging.info(f"Github id: {github_id} id: {id} type: code, successfully. Return code: {returncode}")
            return True
        else:
            logging.info(f"Github id: {github_id} id: {id} type: code, Error. Return code: {returncode}")
            return False
    except subprocess.TimeoutExpired:
        logging.info(f"Github id: {github_id} id: {id} type: code, successfully. Return code: Timeout>{EXEC_TIME}")
        terminate_process_tree(sub.pid)
        return True

def run_bash_script(repo_output: str, id: int, github_id: int, path: str, conda_env: str) -> bool:
    try:
        cmd = f"cd {path}; source activate {conda_env}; {repo_output}"
        sub = subprocess.Popen(["bash", "-c", cmd], stdout=subprocess.PIPE)
        
        timeout = 4 if github_id == 5 else EXEC_TIME
        return_code = sub.wait(timeout=timeout)
        
        if return_code == 0:
            logging.info(f"Github id: {github_id} id: {id} type: script, successfully. Return code: {return_code}")
            return True
        else:
            logging.info(f"Github id: {github_id} id: {id} type: script, Error. Return code: {return_code}")
            return False
    except subprocess.TimeoutExpired:
        logging.info(f"Github id: {github_id} id: {id} type: script, successfully. Return code: Timeout>{EXEC_TIME}")
        terminate_process_tree(sub.pid)
        return True

def validate_arguments(output: str, arguments: Dict[str, Any]) -> bool:
    for argument in arguments:
        if str(arguments[argument]).lower() not in output.lower():
            subprocess.run(['echo', "parameter wrong"])
            return False
    return True

def validate_output_type(output: str, out_type: str) -> bool:
    if out_type == 'Bash Script':
        return '.sh ' in output or '.py ' in output
    else:
        return '.sh ' not in output and '.py ' not in output

def execute_output(output: str, id: int, github_id: int, path: str, conda_env: str, type: str) -> bool:
    if '.sh ' in output or '.py ' in output:
        return run_bash_script(output, id, github_id, path, conda_env)
    else:
        return run_python_code(output, id, github_id, path, conda_env)

def assess_output(outputs: List[str], path: str, conda_env: str, arguments: Dict[str, Any], out_type: str, github_id: int, id: int) -> Tuple[int, int, int]:
    subprocess.run(['echo', f"github_id: {github_id}, id: {id}"])
    subprocess.run(['echo', '--------------------------------------------------------------'])
    
    res = [0, 0, 0]
    dyn = 0
    
    for process_point, output in enumerate(outputs):
        subprocess.run(['echo', f'try {process_point} times'])
        if validate_arguments(output, arguments) and validate_output_type(output, out_type) and execute_output(output, id, github_id, path, conda_env, out_type):
            subprocess.run(['echo', '-- done -- done -- done -- done -- done -- done --'])
            if process_point == 0:
                res = [1, 1, 1]
                dyn = 1
            elif process_point == 1 and dyn != 1:
                res = [0, 1, 1]
                dyn = 1
            elif process_point <= 4 and dyn != 1:
                res = [0, 0, 1]
                dyn = 1
        else:
            subprocess.run(['echo', f"#########################################################"])
    
    return res[0], res[1], res[2]

def align_results_with_dataset(input_path: str, dataset_path: str, split: str) -> List[Dict[str, Any]]:
    with open(input_path, 'r') as fr:
        test_list = [json.loads(data) for data in fr]
    
    ground_truth_list = DatasetDict.load_from_disk(dataset_path)[split]
    
    result_list = []
    for test_data in test_list:
        for data in ground_truth_list:
            if test_data['id'] == data['id'] and test_data['github_id'] == data['github_id']:
                test_data.update({
                    'path': data['path'],
                    'arguments': data['arguments'],
                    'type': data['type']
                })
                result_list.append(test_data)
    return result_list

def compute_repo_pass_rates(json_list: List[Dict[str, Any]]) -> Tuple[Dict[str, List[int]], Dict[str, int]]:
    pass_count = {}
    repo_count = {}
    for data in json_list:
        id_value = data["github_id"]
        repo_path = ID2PATH[id_value]
        if repo_path not in pass_count:
            pass_count[repo_path] = [0, 0, 0]
            repo_count[repo_path] = 0
        
        pass_count[repo_path][0] += int(data["pass1"])
        pass_count[repo_path][1] += int(data["pass2"])
        pass_count[repo_path][2] += int(data["pass5"])
        repo_count[repo_path] += 1
    
    for idx, data in enumerate(pass_count):
        print(f"{ID2PATH[idx+1]} {np.array(pass_count[data]) / repo_count[data]}")
    return pass_count, repo_count

# Main execution
def main() -> None:
    parser = argparse.ArgumentParser(description="Please choose input_path and dataset_path. The result will be written to a derived result_path")
    parser.add_argument('--input_path', type=str, required=True, help="The result path of model output")
    parser.add_argument('--dataset_path', type=str, required=True, help="HuggingFace cached dataset path of ML-Bench")
    parser.add_argument('--split', type=str, default='full', help="The split of the dataset, e.g., 'full' or 'quarter'")
    args = parser.parse_args()
    input_path = args.input_path
    dataset_path = args.dataset_path
    split = args.split
    
    start_time = time.time()
    dt_object = datetime.fromtimestamp(start_time)
    timestamp = dt_object.strftime("%Y%m%d_%H%M%S")
    
    # Create output subdirectories in the same directory as input
    input_dir = os.path.dirname(input_path)
    input_filename = os.path.basename(input_path)
    input_name = os.path.splitext(input_filename)[0]
    
    subdir = os.path.join(input_dir, f"{input_name}{timestamp}")
    os.makedirs(subdir, exist_ok=True)
    
    # Derive result_path and error_path
    result_path = os.path.join(subdir, f"result.jsonl")
    error_path = os.path.join(subdir, f"error.jsonl")
    
    ground_truth_data = align_results_with_dataset(input_path, dataset_path, split)
    ground_truth_data = filter_valid_data(ground_truth_data, error_path)
    
    with open(result_path, "w", encoding="utf-8") as file:
        json.dump({"input_file": input_path, "start_time": dt_object.strftime("%Y-%m-%d %H:%M:%S")}, file)
        file.write("\n")
    
    pass1_list, pass2_list, pass5_list, json_list = [], [], [], []
    
    for data in tqdm(ground_truth_data, desc="Processing data"):
        id, github_id = data["id"], data["github_id"]
        outputs, conda_env = data["output"], ID2CONDA[github_id]
        path = './' + ID2PATH[github_id] + data["path"][1:]
        
        pass1, pass2, pass5 = assess_output(outputs, path, conda_env, data['arguments'], data['type'], github_id, id)
        
        logging.info(f"Github id: {github_id} id: {id} pass1: {pass1} pass2: {pass2} pass5: {pass5}")
        
        pass1_list.append(pass1)
        pass2_list.append(pass2)
        pass5_list.append(pass5)
        
        json_data = {
            "id": id,
            "github_id": github_id,
            "pass1": pass1,
            "pass2": pass2,
            "pass5": pass5
        }
        json_list.append(json_data)
        
        with open(result_path, "a", encoding="utf-8") as file:
            json.dump(json_data, file)
            file.write("\n")
    
    data_counts = len(pass1_list)
    eval_pass1 = sum(pass1_list) / data_counts
    eval_pass2 = sum(pass2_list) / data_counts
    eval_pass5 = sum(pass5_list) / data_counts
    
    json_data = {
        "counts": data_counts,
        "eval_pass1": eval_pass1,
        "eval_pass2": eval_pass2,
        "eval_pass5": eval_pass5,
        "input_file": input_path
    }
    
    with open(result_path, "a", encoding="utf-8") as file:
        json.dump(json_data, file)
        file.write("\n")
    
    pass_count, repo_count = compute_repo_pass_rates(json_list)
    
    with open(result_path, "a", encoding="utf-8") as file:
        for idx, data in enumerate(pass_count):
            json.dump({
                'repo': data,
                'result': list(np.array(pass_count[data]) / repo_count[data])
            }, file)
            file.write("\n")
        file.write("===========================\n")
    
    end_time = time.time()
    print(f'exec time = {end_time - start_time}')
    print(f'Results written to: {result_path}')

if __name__ == "__main__":
    main()

