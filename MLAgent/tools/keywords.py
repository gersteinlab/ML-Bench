import re
from tools.read_yml import read_yaml_file
import json

def get_keywords(query,model_name,api_type,function_type):
    if api_type == "openai":
        from tools.call_openai import call_GPT
    elif api_type == "azure":
        from tools.call_azure import call_GPT
    function_file = "./functions/step1_function.yml"
    function_prompt, function = read_yaml_file(function_file)
    function_prompt = function_prompt.format(query)
    response = call_GPT(function_prompt,model_name,function_type,function)
    print(response)
    function_call_message = response["choices"][0]["message"]["function_call"]
    function_call_json = json.loads(json.dumps(function_call_message.to_dict()))
    res_keywords = json.loads(function_call_json["arguments"])["keywords"]
    keywords = res_keywords.split(', ')
    return keywords



