import json

data_a = []
with open('test_quarter_v4.jsonl', 'r') as file_a:
    for line in file_a:
        try:
            item = json.loads(line)
            data_a.append(item)
        except json.JSONDecodeError:
            print(f"Error decoding JSON in file A: {line}")

data_b = []
with open('test_v4.jsonl', 'r') as file_b:
    for line in file_b:
        try:
            item = json.loads(line)
            data_b.append(item)
        except json.JSONDecodeError:
            print(f"Error decoding JSON in file B: {line}")

filtered_data_a = [item for item in data_a if (str(item['github_id']), str(item['id'])) not in [(b['github_id'], b['id']) for b in data_b]]

with open('filtered_fileF.jsonl', 'w') as filtered_file_a:
    for item in filtered_data_a:
        filtered_file_a.write(json.dumps(item) + '\n')
