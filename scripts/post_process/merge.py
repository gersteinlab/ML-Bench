import json
import tqdm
import os
import pandas as pd
import tqdm
import nltk
from rank_bm25 import BM25Okapi as BM25
from nltk.tokenize import sent_tokenize
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset, Dataset, DatasetDict

nltk.download('punkt')

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


def get_bm25_readme(query, readme, n_sentence=10, max_n_sentence=1000):
    windows = []
    doc_sent = sent_tokenize(readme)[:max_n_sentence]
    for i in range(0, len(doc_sent),  1):
        windows.append(doc_sent[i:i+n_sentence])
    
    docs = [' '.join(window) for window in windows]
    docs = [doc.split() for doc in docs]
    
    bm25 = BM25(docs)
    return ' '.join(bm25.get_top_n(query.split(), docs, n=1)[0])


def process_item(index, data_item, log_dir='../logs', n_sentence=10, max_n_sentence=1000):
    instruction = data_item['instruction']
    readme = os.path.join(log_dir, f"{REPO_2_TXT[data_item['github']]}.txt")

    with open(readme, 'r') as fr:
        readme_content = fr.read()

    data_item['readme_content'] = readme_content
    data_item['bm25_result'] = get_bm25_readme(instruction, readme_content, n_sentence=n_sentence, max_n_sentence=max_n_sentence)
    return data_item


def main(splits, log_dir='../logs', cache_dir='../cache', max_workers=None, n_sentence=10, max_n_sentence=1000):
    os.makedirs(cache_dir, exist_ok=True)
    data = load_dataset('super-dainiu/ml-bench')
    print(f"Loaded dataset from super-dainiu/ml-bench.")

    dataset = DatasetDict()
    for split in splits:
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_item, i, data[split][i], log_dir): i for i in range(len(data[split]))}
            for future in tqdm.tqdm(as_completed(futures), total=len(data[split]), desc=f"Processing {split}"):
                result = future.result()
                if result is not None:
                    results.append(result)

        dataset[split] = Dataset.from_pandas(pd.DataFrame(results))

    dataset.save_to_disk(os.path.join(cache_dir, 'ml-bench-merged'))
    print(f"Dataset saved to {os.path.join(cache_dir, 'ml-bench-merged')}.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--splits', type=str, nargs='+', default=['full', 'quarter'], help='Dataset splits to process')
    parser.add_argument('--max_workers', type=int, default=None, help='Maximum number of worker threads to use')
    parser.add_argument('--log_dir', type=str, default='../logs', help='Directory to store log files')
    parser.add_argument('--cache_dir', type=str, default='../cache', help='Directory to store cache files')
    parser.add_argument('--n_sentence', type=int, default=10, help='Number of sentences to consider in the README')
    parser.add_argument('--max_n_sentence', type=int, default=1000, help='Maximum number of sentences to consider in the README')
    args = parser.parse_args()
    main(splits=args.splits, log_dir=args.log_dir, cache_dir=args.cache_dir, max_workers=args.max_workers, n_sentence=args.n_sentence, max_n_sentence=args.max_n_sentence)
