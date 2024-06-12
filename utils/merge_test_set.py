import pdb
import json
import tqdm
import pandas as pd
from rank_bm25 import BM25Okapi as BM25
from nltk.tokenize import sent_tokenize
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset

REPO_2_TXT = {
    'https://github.com/dmlc/dgl': 'dgl', 
    'https://github.com/google-research/bert': 'bert', 
    'https://github.com/facebookresearch/esm': 'esm', 
    'https://github.com/eriklindernoren/PyTorch-GAN': 'PyTorch-GAN', 
    'https://github.com/salesforce/lavis': 'lavis', 
    'https://github.com/xmu-xiaoma666/External-Attention-pytorch': 'External-Attention-pytorch', 
    'https://github.com/deep-floyd/if': 'if', 
    'https://github.com/NVIDIA/vid2vid': 'vid2vid', 
    'https://github.com/mlfoundations/open_clip': 'open_clip', 
    'https://github.com/thuml/Time-Series-Library': 'Time-Series-Library', 
    'https://github.com/huggingface/pytorch-image-models': 'pytorch-image-models', 
    'https://github.com/vinits5/learning3d': 'learning3d', 
    'https://github.com/microsoft/muzic': 'muzic', 
    'https://github.com/IDEA-Research/Grounded-Segment-Anything': 'Grounded-Segment-Anything', 
}

def get_bm25_readme(query, readme, n_sentence=10):
    windows = []
    doc_sent = sent_tokenize(readme)
    for i in range(0, len(doc_sent),  1):
        windows.append(doc_sent[i:i+n_sentence])
    
    docs = [' '.join(window) for window in windows]
    docs = [doc.split() for doc in docs]
    
    bm25 = BM25(docs)
    return ' '.join(bm25.get_top_n(query.split(), docs, n=1)[0])

def process_item(index, data_item):
    try:
        instruction = data_item['instruction']
        readme = REPO_2_TXT[data_item['github']] + '.txt'
        
        with open(readme, 'r') as fr:
            readme_content = fr.read()
        
        data_item['readme_content'] = readme_content
        data_item['bm25_result'] = get_bm25_readme(instruction, readme_content, n_sentence=10)
        return data_item
    except Exception as e:
        github_id = data_item["github_id"]
        repo_id = data_item["repo_id"]
        # pdb.set_trace()
        print(f"debug github_id {github_id}")
        print(f"debug repo_id {repo_id}")
        print(f"Exception: {e}")
        return None

# 

def main(max_workers=None, split='full'):
    # data = []
    # with open('ML_bench_full_0612.jsonl', 'r') as fp:
    #     for line in fp.readlines():
    #         data.append(json.loads(line))
            
    data = load_dataset('super-dainiu/ml-bench', split=split)

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_item, i, data[i]): i for i in range(len(data))}
        for future in tqdm.tqdm(as_completed(futures), total=len(data)):
            result = future.result()
            if result is not None:
                results.append(result)

    with open('merged' + split + '_benchmark.jsonl', 'w') as fp:
        for line in results:
            json.dump(line, fp)
            fp.write('\n')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--max_workers', type=int, default=None, help='Maximum number of worker threads to use')
    parser.add_argument('--split', type=str, default=None, help='Maximum number of worker threads to use')
    args = parser.parse_args()
    main(max_workers=args.max_workers, split=args.split)
