from datasets import load_dataset, concatenate_datasets
from transformers import CodeLlamaTokenizer
from collections import defaultdict
import pandas as pd
import tiktoken

# Initialize tokenizers
gpt_enc = tiktoken.get_encoding("cl100k_base")
codellama_tokenizer = CodeLlamaTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")

# Function to calculate average token lengths
def calculate_avg_token_lengths(dataset):
    gpt_token_lengths = defaultdict(list)
    codellama_token_lengths = defaultdict(list)
    
    for entry in dataset:
        repo_id = entry['github'].split('/')[-1]
        output = entry['output']
        entry_type = entry['type']
        
        # Tokenize output with GPT (tiktoken) tokenizer
        gpt_tokens = gpt_enc.encode(output)
        gpt_token_lengths[(repo_id, entry_type)].append(len(gpt_tokens))
        
        # Tokenize output with CodeLlama tokenizer
        codellama_tokens = codellama_tokenizer.encode(output)
        codellama_token_lengths[(repo_id, entry_type)].append(len(codellama_tokens))
    
    gpt_avg_lengths = {key: sum(lengths) / len(lengths) for key, lengths in gpt_token_lengths.items()}
    codellama_avg_lengths = {key: sum(lengths) / len(lengths) for key, lengths in codellama_token_lengths.items()}
    
    return gpt_avg_lengths, codellama_avg_lengths

# Load datasets
full_dataset = load_dataset("super-dainiu/ml-bench")['full']
quarter_dataset = load_dataset("super-dainiu/ml-bench")['quarter']
ood_train_dataset = load_dataset("super-dainiu/ml-bench")['ood_train']
id_train_dataset = load_dataset("super-dainiu/ml-bench")['id_train']

# Calculate token lengths for each dataset
gpt_full, codellama_full = calculate_avg_token_lengths(full_dataset)
gpt_quarter, codellama_quarter = calculate_avg_token_lengths(quarter_dataset)
gpt_ood_id_train, codellama_ood_id_train = calculate_avg_token_lengths(concatenate_datasets([ood_train_dataset, id_train_dataset]))

# Combine results into a DataFrame for easy viewing
data = []
all_repos = set([key[0] for key in gpt_full.keys()] + 
                [key[0] for key in gpt_quarter.keys()] + 
                [key[0] for key in gpt_ood_id_train.keys()])
all_types = set([key[1] for key in gpt_full.keys()] + 
                [key[1] for key in gpt_quarter.keys()] + 
                [key[1] for key in gpt_ood_id_train.keys()])

for repo in all_repos:
    for entry_type in all_types:
        row = {
            'Repository': repo,
            'Type': entry_type,
            'Train (GPT Avg Token Length)': gpt_ood_id_train.get((repo, entry_type), '-'),
            'Train (CodeLlama Avg Token Length)': codellama_ood_id_train.get((repo, entry_type), '-'),
            'Test Set (GPT Avg Token Length)': gpt_full.get((repo, entry_type), '-'),
            'Test Set (CodeLlama Avg Token Length)': codellama_full.get((repo, entry_type), '-'),
            '1/4 Test Set (GPT Avg Token Length)': gpt_quarter.get((repo, entry_type), '-'),
            '1/4 Test Set (CodeLlama Avg Token Length)': codellama_quarter.get((repo, entry_type), '-'),
        }
        data.append(row)

df = pd.DataFrame(data)
df.replace('-', pd.NA, inplace=True)

# Calculate averages for each type
type_averages = df.groupby('Type').mean(numeric_only=True)
type_averages['Repository'] = 'Type Average'
type_averages.reset_index(inplace=True)

# Calculate total average
total_average = df.mean(numeric_only=True).to_frame().T
total_average['Repository'] = 'Total Average'
total_average['Type'] = 'Total Average'

# Combine all data
df_final = pd.concat([df, type_averages, total_average], ignore_index=True)
df_final.replace(pd.NA, '-', inplace=True)

# Reorder columns
column_order = ['Repository', 'Type'] + [col for col in df_final.columns if col not in ['Repository', 'Type']]
df_final = df_final[column_order]

# Display the DataFrame
print(df_final)

# Save to CSV
df_final.to_csv('avg_token_lengths_with_types.csv', index=False)