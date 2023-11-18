import sys
sys.path.append('.')
print(sys.path)
from tools.repo_name import get_repo_name
from tools.repo_description import get_repo_description
from tools.read_yml import read_yaml_file
#from tools.get_args import Get_args
from tools.keywords import get_keywords
from tools.repo import get_repo_urls
from tools.generate_index import write_indexfile
from tools.readme import find_readme_files
import subprocess
import os
import shutil
import json5
import json
import argparse
import tiktoken

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def Get_args():
    parser = argparse.ArgumentParser(description="Please choose a model,api_type,function_call to use agent.")
    parser.add_argument('--model_name', type=str, required=True, help="Model name")
    parser.add_argument('--api_type', type=str, required=True, help="Api type")
    parser.add_argument('--function_type', type=str, required=True, help="Function type:auto or none")
    parser.add_argument('--repo_name', type=str, required=True, help="repo_name")
    parser.add_argument('--query', type=str, required=True, help="query")
    parser.add_argument('--id', type=str, required=True, help="id")
    parser.add_argument('--github_id', type=str, required=True, help="github_id")
    args = parser.parse_args()
    model_name = args.model_name
    api_type =args.api_type
    function_type = args.function_type
    repo_name = args.repo_name
    query = args.query
    id = args.id
    github_id = args.github_id
    return model_name,api_type,function_type,repo_name,query,id,github_id
#api_type = "openai"
#model_name = "gpt-3.5-turbo-0613"
#function_type = "auto"
model_name,api_type,function_type,repo_name,query,id,github_id = Get_args()
if api_type == "openai":
    from tools.call_openai import call_GPT
elif api_type == "azure":
    from tools.call_azure import call_GPT

path = "repo/"+repo_name
#step_3 ranking_readmefiles
step3_function_prompt,step3_function = read_yaml_file("functions/step3_function.yml")
write_indexfile(repo_name,path)
indexfile_name = repo_name+"_index.txt"
readme_files = find_readme_files(path)
print(readme_files)
main_readme_path = readme_files[0]
with open (main_readme_path,'r') as file:
    main_readmefile = file.read()
with open(indexfile_name,'r') as f:
    index_file = f.read()
step3_function_prompt = step3_function_prompt.format(repo_name,main_readmefile,query,', '.join(readme_files))
print("step3")
num_tokens = num_tokens_from_string(step3_function_prompt,"cl100k_base")
print(num_tokens)
step3_response = call_GPT(function_prompt = step3_function_prompt ,model_name = model_name,function_type = function_type,function = step3_function)
print(step3_response)
print("*******************************************************************************************************************************************")
print(type(step3_response['choices'][0]['message']['function_call']['arguments']))
print(step3_response['choices'][0]['message']['function_call']['arguments'])
with open("output_log.jsonl","a") as file:
    file.write(str(step3_response['choices'][0]['message']['function_call']['arguments']))
function_args = json5.loads(step3_response['choices'][0]['message']['function_call']['arguments'])
task_flag = function_args["task_flag"]
ranked_readmefiles = function_args["ranked_readmefiles"]
with open("whole_log.log","a") as file:
    three_response = str(step3_response)
    file.write("step3_response:"+three_response+"\n")
print(ranked_readmefiles)
with open("ranked_readmefiles.txt","a") as file:
    file.write(str(ranked_readmefiles))
print(type(ranked_readmefiles))
if isinstance(ranked_readmefiles, str):
    ranked_readmefiles = [s.strip() for s in ranked_readmefiles.split(',')] 
    print(type(ranked_readmefiles))
if task_flag == "No":
    print("This repo cannot do the task")
    sys.exit()
elif task_flag == "Yes":
    #step_4
    for ranked_readmefile in ranked_readmefiles:
        step4_function_prompt,step4_function = read_yaml_file("functions/step4_function.yml")
        with open (ranked_readmefile,'r') as file:
            readmefile = file.read()
        with open(indexfile_name,"r") as f:
            index_file = f.read()
        step4_function_prompt = step4_function_prompt.format(index_file,readmefile,query)
        print("step4")
        num_tokens = num_tokens_from_string(step4_function_prompt,"cl100k_base")
        print(num_tokens)
        step4_response = call_GPT(function_prompt = step4_function_prompt ,model_name = model_name,function_type =function_type ,function = step4_function)
        print("*******************************************************************************************************************************************")
        print(type(step4_response['choices'][0]['message']['function_call']['arguments']))
        print(step4_response['choices'][0]['message']['function_call']['arguments'])
        with open("output_log.jsonl","a") as file:
            file.write(str(step4_response['choices'][0]['message']['function_call']['arguments']))
        function_args = json5.loads(step4_response['choices'][0]['message']['function_call']['arguments'])
        flag = function_args["flag"]
        with open("whole_log.log","a") as file:
            four_response = str(step4_response)
            file.write("step3_response:"+four_response+"\n")        
        if flag == "No":
            continue
        if flag == "CODE":
            code = function_args["code"]
            with open("test_v4.8.jsonl","a") as file:
                data = {
                    "id":id,
                    "github_id":github_id,
                    "output":code            
                    }
                json_line = json.dumps(data)
                file.write(json_line + '\n')
            print(code)
            sys.exit()
        if flag == "FILE":
            call_path = function_args["file_path"]
            path = "repo/"+repo_name+"/"+call_path
            with open (path,"r") as file:
                py_file = file.read()
            args_function_prompt,args_function = read_yaml_file("functions/args_function.yml")
            args_function_prompt = args_function_prompt.format(py_file)
            print("args_code")
            num_tokens = num_tokens_from_string(args_function_prompt,"cl100k_base")
            print(num_tokens) 
            print(args_function_prompt)
            args_response = call_GPT(function_prompt = args_function_prompt ,model_name = model_name ,function_type = function_type,function = args_function)
            print(args_response)
            print("*******************************************************************************************************************************************")
            print(type(args_response['choices'][0]['message']['function_call']['arguments']))
            print(args_response['choices'][0]['message']['function_call']['arguments'])
            with open("output_log.jsonl","a") as file:
                file.write(str(args_response['choices'][0]['message']['function_call']['arguments']))
            function_args = json5.loads(args_response['choices'][0]['message']['function_call']['arguments'])
            args_code = function_args["args_code"]
            with open("whole_log.log","a") as file:
                ags_response = str(args_response)
                file.write("args_response:"+ags_response+"\n")
            code_function_prompt,code_function = read_yaml_file("functions/code_function.yml")
            code_function_prompt = code_function_prompt.format(ranked_readmefile,args_code,query)
            print("step_code")
            num_tokens = num_tokens_from_string(code_function_prompt,"cl100k_base")
            print(num_tokens) 
            print(code_function_prompt)
            code_response = call_GPT(function_prompt = code_function_prompt ,model_name = model_name,function_type = function_type,function = code_function)
            print(code_response)
            print("*******************************************************************************************************************************************")
            print(type(code_response['choices'][0]['message']['function_call']['arguments']))
            print(code_response['choices'][0]['message']['function_call']['arguments'])
            with open("output_log.jsonl","a") as file:
                file.write(str(code_response['choices'][0]['message']['function_call']['arguments']))
            function_args = json5.loads(code_response['choices'][0]['message']['function_call']['arguments'])
            flag_finally =  function_args["flag"]
            with open("whole_log.log","a") as file:
                cod_response = str(code_response)
                file.write("code_response:"+cod_response+"\n")
                file.write("----end----"+"\n")
            if flag_finally == "CODE":
                finally_code = function_args["finally_code"]
                print(finally_code)
                with open("test_v4.8.jsonl","a") as file:
                    data = {
                        "id":id,
                        "github_id":github_id,
                        "output":finally_code            
                        }
                    json_line = json.dumps(data)
                    file.write(json_line + '\n')
                sys.exit()
                break
            if flag_finally =="No":
                continue
    with open("test_v4.8_pass1.jsonl","a") as file:
        data = {
                "id":id,
                "github_id":github_id,
                "output":""            
        }
        json_line = json.dumps(data)
        file.write(json_line + '\n')  
else:
    print("task_flag error")
    sys.exit()