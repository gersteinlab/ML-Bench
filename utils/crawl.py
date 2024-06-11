import json
import requests
from bs4 import BeautifulSoup

urls = [
 'https://github.com/dmlc/dgl/tree/master/examples/pytorch/arma/README.md',
 'https://github.com/dmlc/dgl/blob/master/examples/pytorch/gcn/README.md',
 'https://github.com/dmlc/dgl/blob/master/examples/pytorch/dtgrnn/README.md',
 'https://github.com/dmlc/dgl/blob/master/examples/pytorch/dagnn/README.md',
 'https://github.com/dmlc/dgl/blob/master/examples/pytorch/capsule/README.md',
 'https://github.com/dmlc/dgl/blob/master/examples/pytorch/correct_and_smooth/README.md',
 'https://github.com/dmlc/dgl/blob/master/examples/pytorch/dgi/README.md',
 'https://github.com/dmlc/dgl/tree/master/examples/pytorch/NGCF/README.md',
 'https://github.com/dmlc/dgl/blob/master/examples/pytorch/GATNE-T/README.md',
 'https://github.com/dmlc/dgl/blob/master/examples/pytorch/cluster_gcn/README.md',
 'https://github.com/dmlc/dgl/blob/master/examples/pytorch/dgmg/README.md',
 'https://github.com/dmlc/dgl/blob/master/examples/pytorch/appnp/README.md',
 'https://github.com/dmlc/dgl/blob/master/examples/pytorch/eeg-gcnn/README.md',
 'https://github.com/dmlc/dgl/tree/master/examples/pytorch/eges/README.md',
 'https://github.com/dmlc/dgl/tree/master/examples/pytorch/compGCN/README.md',
 'https://github.com/dmlc/dgl/tree/master/examples/pytorch/bgrl/README.md',
 'https://github.com/dmlc/dgl/tree/master/examples/pytorch/gat/README.md',
 'https://github.com/dmlc/dgl/blob/master/examples/pytorch/caregnn/README.md',
 'https://github.com/google-research/bert/blob/master/README.md',
 'https://github.com/salesforce/LAVIS/blob/main/README.md',
 'https://github.com/deep-floyd/IF/blob/develop/README.md',
 'https://github.com/NVIDIA/vid2vid/blob/master/README.md',
 'https://github.com/facebookresearch/esm/blob/master/README.md',
 'https://github.com/mlfoundations/open_clip/blob/main/README.md',
 'https://github.com/thuml/Time-Series-Library/blob/main/README.md',
 'https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/README_EN.md'
 ]


for url in urls:
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        readme_content = soup.find("article").text
        # print(readme_content)
        with open("readme_content.txt","r") as file:
            file.write(readme_content)
            
    else:
        print(f"Failed to retrieve the page. Status code: {response.status_code}")
