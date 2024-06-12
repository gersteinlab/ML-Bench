# !pip install rank_bm25 nltk
# Author: Daniel Shao
import json
import pandas as pd
from rank_bm25 import BM25Okapi as BM25
from nltk.tokenize import sent_tokenize

def get_bm25_readme(query, readme, n_sentence=10):
    
    windows = []
    doc_sent = sent_tokenize(readme)
    for i in range(0, len(doc_sent),  1):
        windows.append(doc_sent[i:i+n_sentence])
    
    docs = [' '.join(window) for window in windows]
    docs = [doc.split() for doc in docs]
    
    bm25 = BM25(docs)
    return ' '.join(bm25.get_top_n(query.split(), docs, n=1)[0])


data = []
with open('path_to_dataset.jsonl', 'r') as fp:
    for line in fp.readlines():
        data.append(json.loads(line))
        cur_data = json.loads(line)
        


for index in range(len(data)):
    try:
        instruction = data[index]['instructions']
        readme = data[index]['readme']


        data[index]['bm25_result'] = get_bm25_readme(instruction, readme, n_sentence=10)
    except:
        github_id = data[index]["github_id"]
        repo_id = data[index]["repo_id"]
        print(f"debug github_id {github_id}")
        print(f"debug repo_id {repo_id}")

with open('path_to_save_bm25result.jsonl', 'w') as fp:
    for line in data:
        json.dump(line, fp)
        fp.write('\n')
