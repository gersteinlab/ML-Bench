# ML-AGENT-1.0

## Start without specifying the repo
python run.py --model_name MODEL_NAME --api_type API_TYPE --function_type FUNCTION_TYPE

MODEL_NAME:Set your key and model in the config folder
API_TYPE:azure,openai
FUNCTION_TYPE:auto,none



## Start with specifying the repo
python run_choose_repo.py --model_name MODEL_NAME --api_type API_TYPE --function_type FUNCTION_TYPE --repo_name REPO_NAME --query QUERY --id ID --github_id GITHUB_ID

MODEL_NAME:Set your key and model in the config folder
API_TYPE:azure,openai
FUNCTION_TYPE:auto,none
REPO_NAME:One of the 10 repos we offer
QUERY:User instruction
GITHUB_ID:The id corresponding to the repo in our dataset

## More information and the latest ML-AGENT code will be update soon