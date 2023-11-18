import sys
sys.path.append('.')
print(sys.path)
from tools.repo_name import get_repo_name
from tools.repo_description import get_repo_description
from tools.read_yml import read_yaml_file
from tools.get_args import Get_args
from tools.keywords import get_keywords
from tools.repo import get_repo_urls
from tools.generate_index import write_indexfile
from tools.readme import find_readme_files

import subprocess
import os
import shutil
import json
import json5
#api_type = "openai"
#model_name = "gpt-3.5-turbo-0613"
#function_type = "auto"

model_name,api_type,function_type = Get_args()
query = input("Please input your query:")
if api_type == "openai":
    from tools.call_openai import call_GPT
elif api_type == "azure":
    from tools.call_azure import call_GPT

#step_1 get keywords and repo_urls
keywords = get_keywords(query=query,model_name = model_name,api_type= api_type,function_type = function_type)
repo_urls = get_repo_urls(query=query,model_name = model_name,api_type= api_type,function_type = function_type)
print(repo_urls)
request_args = []
#step_2 Select one from the list of repo_urls
for url in repo_urls:
    repo_name = get_repo_name(url)
    repo_description = get_repo_description(url)
    request_repo = "repo_name:"+repo_name+"|"+"repo_description:"+repo_description+"|"+"repo_url:"+"|"+url
    request_args.append(request_repo)
step2_function_prompt,step2_function = read_yaml_file("functions/step2_function.yml")
print(step2_function_prompt,step2_function)
step2_function_prompt = step2_function_prompt.format(keywords,', '.join(repo_urls))
print(step2_function_prompt)
step2_response = call_GPT(function_prompt = step2_function_prompt ,model_name = model_name,function_type = function_type,function = step2_function)
print(step2_response)
function_call_message = step2_response["choices"][0]["message"]["function_call"]
function_call_json = json.loads(json.dumps(function_call_message.to_dict()))
ranked_array = json.loads(function_call_json["arguments"])["ranked_array"]
print(ranked_array)
print(type(ranked_array))
if isinstance(ranked_array, str):
    ranked_array =ranked_array.split(',')
    print(type(ranked_array))

#step_3 ranking_readmefiles
for repo_url in ranked_array:
    step3_function_prompt,step3_function = read_yaml_file("functions/step3_function.yml")
    repo_name = get_repo_name(repo_url)
    path = repo_name+"/"
    if os.path.exists(path):
        shutil.rmtree(path)
    request_query = "git clone"+" "+repo_url+" "+repo_name
    try:
        subprocess.check_call(request_query ,shell=True)
    except subprocess.CalledProcessError as e:
        print("error:", e)
    write_indexfile(repo_name,path)
    indexfile_name = repo_name+"_index.txt"
    readme_files = find_readme_files(repo_name)
    main_readme_path = readme_files[0]
    with open (main_readme_path,'r') as file:
        main_readmefile = file.read()
    with open(indexfile_name,'r') as f:
        index_file = f.read()
    step3_function_prompt = step3_function_prompt.format(repo_name,index_file,main_readmefile,query,', '.join(readme_files))
    step3_response = call_GPT(function_prompt = step3_function_prompt ,model_name = model_name,function_type = function_type,function = step3_function)
    print(step3_response['choices'][0]['message']['function_call']['arguments'])
    function_call_message = step3_response["choices"][0]["message"]["function_call"]
    function_call_json = json.loads(json.dumps(function_call_message.to_dict()))
    task_flag = json.loads(function_call_json["arguments"])["task_flag"]
    ranked_readmefiles = json.loads(function_call_json["arguments"])["ranked_readmefiles"]
    if isinstance(ranked_readmefiles, str):
        ranked_readmefiles = ranked_readmefiles.split(',')
        print(type(ranked_readmefiles))
    print(ranked_readmefiles)
    if task_flag == "No":
        continue
    elif task_flag == "Yes":
        #step_4
        for ranked_readmefile in ranked_readmefiles:
            step4_function_prompt,step4_function = read_yaml_file("functions/step4_function.yml")
            with open (ranked_readmefile,'r') as file:
                readmefile = file.read()
            with open(indexfile_name,"r") as f:
                index_file = f.read()
            step4_function_prompt = step4_function_prompt.format(index_file,readmefile,query)
            step4_response = call_GPT(function_prompt = step4_function_prompt ,model_name = model_name,function_type =function_type ,function = step4_function)
            function_call_message = step4_response["choices"][0]["message"]["function_call"]
            function_call_json = json.loads(json.dumps(function_call_message.to_dict()))
            flag = json.loads(function_call_json["arguments"])["flag"]
            if flag == "No":
                continue
            if flag == "CODE":
                code = json.loads(function_call_json["arguments"])["code"]
                print(code)
                sys.exit()
            if flag == "FILE":
                call_path = json.loads(function_call_json["arguments"])["file_path"]
                path = repo_name+"/"+call_path
                with open (path,"r") as file:
                    py_file = file.write()
                args_function_prompt,args_function = read_yaml_file("functions/args_function.yml")
                args_function_prompt = args_function_prompt.format(py_file)
                args_response = call_GPT(function_prompt = args_function_prompt ,model_name = model_name ,function_type = function_type,function = args_function)
                function_call_message = args_response["choices"][0]["message"]["function_call"]
                function_call_json = json.loads(json.dumps(function_call_message.to_dict()))
                args_code = json.loads(function_call_json["arguments"])["args_code"]
                code_function_prompt,code_function = read_yaml_file("functions/args_function.yml")
                code_function_prompt = code_function_prompt.format(anked_readmefile,args_code,query)
                code_response = call_GPT(function_prompt = code_function_prompt ,model_name = model_name,function_type = function_type,function = code_function)
                function_call_message = code_response["choices"][0]["message"]["function_call"]
                function_call_json = json.loads(json.dumps(function_call_message.to_dict()))
                flag_finally =  json.loads(function_call_json["arguments"])["flag"]
                if flag_finally == "CODE":
                    finally_code = json.loads(function_call_json["arguments"])["finally_code"]
                    print(finally_code)
                    sys.exit()
                    break
                if flag_finally =="No":
                    continue
    else:
        continue