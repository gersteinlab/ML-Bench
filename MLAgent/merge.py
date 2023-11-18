import json

merged_data = {}

file_names = ['new_file_4.0.jsonl', 'new_file_4.6.jsonl', 'new_file_4.7.jsonl', 'new_file_4.8.jsonl', 'new_file_4.9.jsonl']

for file_name in file_names:
    with open(file_name, 'r') as file:
        for line in file:
            data = json.loads(line)

            id_value = data['id']
            github_id_value = data['github_id']

            if (id_value, github_id_value) in merged_data:
                merged_data[(id_value, github_id_value)]['output'].append(data['output'])
            else:
                merged_data[(id_value, github_id_value)] = {
                    'id': id_value,
                    'github_id': github_id_value,
                    'output': [data['output']]
                }

with open('merged_output.jsonl', 'w') as merged_file:
    for merged_item in merged_data.values():
        merged_item['output'] = sum(merged_item['output'], [])
        merged_file.write(json.dumps(merged_item) + '\n')
