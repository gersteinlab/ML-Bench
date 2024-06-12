import yaml

def read_yaml_file(yaml_file_path):
    try:
        with open(yaml_file_path, 'r') as file:
            data = yaml.safe_load(file)

        function_prompt = data.get('function_prompt', '')
        function = data.get('function', {})

        return function_prompt, function
    except Exception as e:
        return None, None