import openai
import yaml
import os


def call_GPT(function_prompt,model_name,function_type,function):
    if function_type == "auto":
        with open("./config/config_azure.yml", "r") as yaml_file:
            config = yaml.safe_load(yaml_file)
        openai.api_base = config["api_base"]
        openai.api_type = config["api_type"]
        openai.api_version = config["api_version"]
        #openai.api_proxy = config["api_proxy"]
        openai.api_key = config["openai_keys"][model_name][0]["api_key"]
        try:
            res = openai.ChatCompletion.create(
                        engine=model_name,
                        messages=[
                            {"role": "user",
                             "content": function_prompt}
                        ], 
                        functions = [function],
                        function_call = "auto" ,
                    )
            return res
        except Exception as e:
            print("An exception occurred:", e)
    elif function_type == "none":
        with open("./config/config_azure.yml", "r") as yaml_file:
            config = yaml.safe_load(yaml_file)
        openai.api_base = config["api_base"]
        openai.api_type = config["api_type"]
        openai.api_version = config["api_version"]
        #openai.api_proxy = config["api_proxy"]
        openai.api_key = config["openai_keys"][model_name][0]["api_key"]
        try:
            res = openai.ChatCompletion.create(
                        engine=model_name,
                        messages=[
                            {"role": "user",
                             "content": function_prompt}
                        ]
                    )
            return res
        except Exception as e:
            print("An exception occurred:", e)

