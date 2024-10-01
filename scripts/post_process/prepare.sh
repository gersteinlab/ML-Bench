#!/bin/bash

REPO_DIR=repos
LOG_DIR=logs
CACHE_DIR=cache

REPO_LIST=(
    "bert:eedf5716ce1268e56f0a50264a88cafad334ac61"
    "dgl:f5c8c0b500db810210d22add9b7d6fc809edc0d2"
    "esm:2b369911bb5b4b0dda914521b9475cad1656b2ac"
    "External-Attention-pytorch:90e774ed4a2a734fe91dde6a4a737de0f0c54122"
    "Grounded-Segment-Anything:d1c510118c0fdffb87bab3cd3dfe3e6b02a36f8b"
    "if:ffc8163891682beaca0c5eb6b9077860a3bc6509"
    "lavis:7f00a08"
    "learning3d:becc4a0017488390d3f04e4bc2ce9efb368115c5"
    "muzic:9d52ff30b209a19fe51606242797c51a1bf7d7c1"
    "open_clip:91923dfc376afb9d44577a0c9bd0930389349438"
    "PyTorch-GAN:36d3c77e5ff20ebe0aeefd322326a134a279b93e"
    "pytorch-image-models:205d8ad37c838c20c541b787dca840edd9ed1408"
    "Time-Series-Library:51e80b6f76b7823fea2512d6d170b1b83b149d2d"
    "vid2vid:2e6d13755fc2e33200e7d4c0c44f2692d6ab0898"
)

mkdir -p $LOG_DIR
mkdir -p $CACHE_DIR

for item in "${REPO_LIST[@]}"; do
    repo=$(echo $item | cut -d':' -f1)
    commit=$(echo $item | cut -d':' -f2)
    python scripts/post_process/crawl_repos.py $REPO_DIR/$repo $CACHE_DIR/$repo.txt $commit
done

python scripts/post_process/merge.py --split full quarter --cache_dir $CACHE_DIR --max_workers 40
