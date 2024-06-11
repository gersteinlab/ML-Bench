import argparse

def Get_args():
    parser = argparse.ArgumentParser(description="Please choose a model,api_type,function_call to use agent.")
    parser.add_argument('--model_name', type=str, required=True, help="Model name")
    parser.add_argument('--api_type', type=str, required=True, help="Api type")
    parser.add_argument('--function_type', type=str, required=True, help="Function type:auto or none")
    args = parser.parse_args()
    model_name = args.model_name
    api_type =args.api_type
    function_type = args.function_type
    return model_name,api_type,function_type
