import json
import subprocess
import subprocess
import logging
import psutil
import argparse
from datetime import datetime
import time
import numpy as np
import pdb

logging.basicConfig(filename='exec.log', level=logging.INFO)

EXEC_TIME = 10
LIST_LEN = 5
start_time = time.time()
ID2CONDA = {
        1: "dgl_DS",
        2: "bert_DS", 
        3: "lavis_DS", 
        4: "if_DS", 
        5: "V2V_DS", 
        6: "esm_DS", 
        7: "OP_DS", 
        8: "TSL_DS", 
        9: "EAP_DS",
        10: "PG_DS", 
        11: "PIM_DS", 
        13: "L3_DS", 
        14: "MZ2_DS", 
        15: "GSA2_DS" 
        }

ID2PATH = {
        1: "dgl", 
        2: "bert", 
        3: "lavis", 
        4: "if", 
        5: "vid2vid",
        6: "esm", 
        7: "open_clip", 
        8: "Time-Series-Library", 
        9: "External-Attention-pytorch", 
        10: "PyTorch-GAN",
        11: "pytorch-image-models", 
        13: "learning3d", 
        14: "muzic", 
        15: "Grounded-Segment-Anything" }
def kill_process_tree(pid):
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


def eval_data(GT_datas:list)->list:
        correct_datas = []
        uncorrect_datas = []
        count = 0
        for data in GT_datas:
                lenth = len(data["output"])
                if 'path' not in data:
                        pdb.set_trace()
                if lenth == LIST_LEN:
                        correct_datas.append(data)
                else:
                        count = count + 1
                        logging.info(str(data)+" is error and it will be delete. Error json data will be writen in error_data.jsonl.")
                        print(str(data)+" is error and it will be delete. Error json data will be writen in error_data.jsonl.")
                        uncorrect_datas.append(data)
        with open("error_data.jsonl","a",encoding="utf-8") as file:
                file.write("Error data counts is "+str(count)+"\n")
                logging.info("Error data counts is "+str(count)+"\n")
                json.dump(uncorrect_datas,file)
        return correct_datas

def exec_code(repo_output:str,id:int,github_id:int,path:str,conda_env:str)->bool:
        with open(path + "/temp.py","w",encoding="utf-8") as file:
               file.write(repo_output)
        file.close()

        
        try:
                # sub = subprocess.Popen("bash -c "+"'"+"cd "+path+";source activate "+conda_env+";conda info --envs;"+"python temp.py"+"'",shell=True)
                sub = subprocess.Popen("bash -c "+"'"+"cd "+path+";source activate "+conda_env+";"+"python temp.py"+"'",shell=True, stdout=subprocess.PIPE) # , stderr=subprocess.PIPE
                if github_id == 5:
                    returncode = sub.wait(timeout=4)
                elif github_id == 11:
                    returncode = sub.wait(timeout=7)
                else:
                    returncode = sub.wait(timeout=EXEC_TIME)
                if returncode == 0:
                       logging.info("Github id: " + str(github_id) + " id: " +  str(id) + " type: code, successfully. Return code: " + str(returncode))
                       return True
                else:
                       logging.info("Github id: " + str(github_id) + " id: " +  str(id) + " type: code, Error. Return code: " + str(returncode))
                       return False
        except subprocess.TimeoutExpired:
               logging.info("Github id: " + str(github_id) + " id: " +  str(id) + " type: code, successfully. "+"Return code：Timeout>10")
               kill_process_tree(sub.pid)
               return True
        
def exec_script(repo_output:str,id:int,github_id:int,path:str,conda_env:str)->bool:
        repo_output = repo_output
        try:
                # sub = subprocess.Popen("bash -c "+"'"+"cd "+path+";source activate "+conda_env+";conda info --envs;"+repo_output+"'",shell=True)
                sub = subprocess.Popen("bash -c "+"'"+"cd "+path+";source activate "+conda_env+";"+repo_output+"'",shell=True, stdout=subprocess.PIPE) # , stderr=subprocess.PIPE
                if github_id != 5:
                    returncode = sub.wait(timeout=EXEC_TIME)
                else:
                    returncode = sub.wait(timeout=4)
                if returncode == 0:
                       logging.info("Github id: " + str(github_id) + " id: " +  str(id) + " type: script, successfully. Return code: " + str(returncode))
                       return True
                else:
                       logging.info("Github id: " + str(github_id) + " id: " +  str(id) + " type: script, Error. Return code: " + str(returncode))
                       return False
        except subprocess.TimeoutExpired:
               logging.info("Github id: " + str(github_id) + " id: " +  str(id) + " type: script, successfully. "+"Return code：Timeout>10")
               kill_process_tree(sub.pid)
               return True

def check_hit(output, arguments):
    
    for argument in arguments:
        if str(arguments[argument]).lower() not in output.lower():
            subprocess.run(['echo', "parameter wrong"])
            return False
    return True

def check_type(output, out_type):
    if out_type == 'Bash Script':
        if '.sh ' in output or '.py ' in output:
            return True
        else: 
            subprocess.run(['echo', "type wrong"])
            return False # False
    else:
        if '.sh ' in output or '.py ' in output:
            subprocess.run(['echo', "type wrong"])
            return False # False
        else:
            return True

def exec_code_or_script(output, id, github_id, path, conda_env, type):
    if '.sh ' in output or '.py ' in output:
        return exec_script(output, id, github_id, path, conda_env)
    else:
        return exec_code(output, id, github_id,path,conda_env)
    

def eval_output(outputs:list,path:str,conda_env:str, arguments:dict, out_type:str, github_id, id)->tuple:
        process_point = 0
        # pdb.set_trace() 
        #
        subprocess.run(['echo', f"github_id: {github_id}, id: {id}"])
        subprocess.run(['echo', '--------------------------------------------------------------'])
        subprocess.run(['echo', '--------------------------------------------------------------'])
        subprocess.run(['echo', '--------------------------------------------------------------'])
        
        res = [0, 0, 0]
        dyn = 0
        
        for output in outputs:
                subprocess.run(['echo', f'try {process_point} times'])
                if process_point == 0:
                    
                        if check_hit(output, arguments) and check_type(output, out_type) and exec_code_or_script(output, id, github_id,path,conda_env, out_type):
                                subprocess.run(['echo', '-- done -- done -- done -- done -- done -- done --'])
                                dyn = 1
                                res = [1, 1, 1]
                        else:
                                subprocess.run(['echo', f"#########################################################"])
                elif process_point == 1:
                        if check_hit(output, arguments) and check_type(output, out_type) and exec_code_or_script(output, id, github_id,path,conda_env, out_type):
                                subprocess.run(['echo', '-- done -- done -- done -- done -- done -- done --'])
                                if dyn != 1:
                                    res = [0, 1, 1]
                                    dyn = 1
                        else:
                                subprocess.run(['echo', f"#########################################################"])
                elif process_point <= 4:
                        if check_hit(output, arguments) and check_type(output, out_type) and exec_code_or_script(output, id, github_id,path,conda_env, out_type):
                                subprocess.run(['echo', '-- done -- done -- done -- done -- done -- done --'])
                                if dyn != 1:
                                    res = [0, 0, 1]
                                    dyn = 1
                        else:
                                subprocess.run(['echo', f"#########################################################"])
                process_point += 1
        # print(process_point)
        return res[0], res[1], res[2]
              
def map_result_to_base_set(result_path, base_path):
    
    test_datas = []
    with open(result_path, 'r') as fr:
        for data in fr.readlines():
            test_datas.append(json.loads(data))
    fr.close()
            
    GT_datas = []
    with open(base_path, 'r') as fr2:
        for data in fr2.readlines():
            GT_datas.append(json.loads(data))
    fr2.close()
    
    result_datas = []
    for test_data in test_datas:
        for GT_data in GT_datas:
            if test_data['id'] == GT_data['id'] and test_data['github_id'] == GT_data['github_id']:
                test_data['path'] = GT_data['path']
                test_data['arguments'] = GT_data['arguments']
                test_data['type'] = GT_data['type']
                result_datas.append(test_data)
    # pdb.set_trace()
    return result_datas


def Get_args():
    parser = argparse.ArgumentParser(description="Please choose input result_path and base_path")
    parser.add_argument('--result_path', type=str, required=True, help="result path")
    args = parser.parse_args()
    result_path = args.result_path
    return result_path

def calculate_repo_pass(json_datas):
    pass_count = {}
    repo_count = {}
    for data in json_datas:
        id_value = data["github_id"]
        pass_value_1 = data["pass1"]
        pass_value_2 = data["pass2"]
        pass_value_5 = data["pass5"]
        if ID2PATH[id_value] not in pass_count:
            pass_count[ID2PATH[id_value]] = []
        if ID2PATH[id_value] not in repo_count:
            repo_count[ID2PATH[id_value]] = 1
        else:
            repo_count[ID2PATH[id_value]] += 1
        if len(pass_count[ID2PATH[id_value]]) < 1:
            pass_count[ID2PATH[id_value]].append(int(pass_value_1))
        else:
            pass_count[ID2PATH[id_value]][0] += int(pass_value_1)
        
        if len(pass_count[ID2PATH[id_value]]) < 2:
            pass_count[ID2PATH[id_value]].append(int(pass_value_2))
        else:
            pass_count[ID2PATH[id_value]][1] += int(pass_value_2)
            
        if len(pass_count[ID2PATH[id_value]]) < 3:
            pass_count[ID2PATH[id_value]].append(int(pass_value_5))
        else:
            pass_count[ID2PATH[id_value]][2] += int(pass_value_5)      
    # pdb.set_trace() 
    for idx, data in enumerate(pass_count):
        # pdb.set_trace()
        print(ID2PATH[idx+1] + ' ' + str(np.array((pass_count[data])) / repo_count[data]))
    return pass_count, repo_count
            
pass1_list = []
pass2_list = []
pass5_list = []
json_datas = []


result_path = Get_args()
base_path = "./merged_full_benchmark.jsonl"   # "./benchmark/ML_bench_quarter.jsonl"
GT_datas = map_result_to_base_set(result_path, base_path)

dt_object = datetime.fromtimestamp(start_time)

with open("eval_total_user.jsonl","a",encoding="utf-8") as file:
        json_line = json.dumps({"input file": result_path, 
                                "start time": dt_object.strftime("%Y-%m-%d %H:%M:%S")})
        file.write(json_line+"\n")

GT_datas = eval_data(GT_datas)
flag = 0
for data in GT_datas:
        
        id = data["id"]
        github_id = data["github_id"]
        outputs = data["output"]
        conda_env = ID2CONDA[github_id]
        path = data["path"]
        new_path = './' + ID2PATH[github_id] + path[1:]
        pass1,pass2,pass5 = eval_output(outputs,new_path,conda_env, data['arguments'], data['type'], github_id, id)
        repo_name = ID2PATH[github_id]
        logging.info("Github id: " + str(github_id) + " id: " +  str(id) + " pass1: " +str(pass1) + " pass2: " +str(pass2) + " pass5: " +str(pass5))
        pass1_list.append(pass1)
        pass2_list.append(pass2)
        pass5_list.append(pass5)
        json_data = {
                        "id":id,
                        "github_id":github_id,
                        "pass1":pass1,
                        "pass2":pass2,
                        "pass5":pass5
                }
        json_datas.append(json_data)
        with open("eval_total_user.jsonl","a",encoding="utf-8") as file:
                json_line = json.dumps(json_data)
                file.write(json_line+"\n")
                
                
data_counts = len(pass1_list)
# pdb.set_trace()
eval_pass1 = sum(pass1_list)/data_counts
eval_pass2 = sum(pass2_list)/data_counts
eval_pass5 = sum(pass5_list)/data_counts
json_data = {
                "counts":data_counts,
                "eval_pass1":eval_pass1,
                "eval_pass2":eval_pass2,
                "eval_pass5":eval_pass5
        }
with open("eval_total_user.jsonl","a",encoding="utf-8") as file:
        json_line = json.dumps(json_data)
        file.write(json_line+"\n")

json_data['input file'] = result_path

with open("eval_result_user.jsonl","a",encoding="utf-8") as file:
        json_line = json.dumps(json_data)
        file.write(json_line+"\n")

pass_count, repo_count = calculate_repo_pass(json_datas)


with open("eval_result_user.jsonl","a",encoding="utf-8") as file:
    for idx, data in enumerate(pass_count):
        # pdb.set_trace()
        json_line = json.dumps({
            'repo': data, 
            'res': list(np.array((pass_count[data])) / repo_count[data])
        })
        file.write(json_line+"\n")

with open("eval_result_user.jsonl","a",encoding="utf-8") as file:
    file.write("===========================\n")

end_time = time.time()

print('exec time = ' + str(end_time - start_time))
