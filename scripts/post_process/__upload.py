from datasets import load_dataset, DatasetDict
import datasets

# Load the original dataset
data = load_dataset("super-dainiu/ml-bench")

# Load new datasets
full_new = load_dataset('json', data_files='full.jsonl')['train']
quarter_new = load_dataset('json', data_files='quarter.jsonl')['train']
ood_train_new = load_dataset('json', data_files='ood_train.jsonl')['train']
id_train_new = load_dataset('json', data_files='id_train.jsonl')['train']

# Function to update the 'type' column
def update_type_column(old_dataset, new_dataset):
    return old_dataset.map(lambda example, idx: {"type": new_dataset[idx]['type']}, with_indices=True)

# Update datasets
data['full'] = update_type_column(data['full'], full_new)
data['quarter'] = update_type_column(data['quarter'], quarter_new)
data['ood_train'] = update_type_column(data['ood_train'], ood_train_new)
data['id_train'] = update_type_column(data['id_train'], id_train_new)

# # Verify the change
print(data['quarter'][-1]['type'], quarter_new[-1]['type'])

data['full'] = full_new
data['quarter'] = quarter_new
data['ood_train'] = ood_train_new
data['id_train'] = id_train_new

# Push to Hub
data.push_to_hub('super-dainiu/ml-bench', commit_message='fix type column in datasets')