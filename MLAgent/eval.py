import subprocess
import json
import os

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
        10: "PyTorch-GAN" }


test_datas = []
with open("filtered_fileF.jsonl","r") as file:
    for data in file.readlines():
            test_datas.append(json.loads(data))

print(len(test_datas))

for test_data in test_datas:
    id = test_data["id"]
    github_id =test_data["github_id"]
    query = test_data["extend instructions"]
    repo_name = ID2PATH[github_id]
    print(id)
    print(github_id)
    str_call = "python "+"run_choose_repo.py "+" --model_name gpt-4-32k"+" --api_type azure"+" --function_type auto"+" --repo_name "+repo_name+" --query "+'"'+query+'"'+" --id "+str(id)+" --github_id "+str(github_id)
    print(str)
    try:
        subprocess.run(str_call,shell = True)
    except:
        print("error")
        continue