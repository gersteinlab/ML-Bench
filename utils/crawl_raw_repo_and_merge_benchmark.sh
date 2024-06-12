python crawl_repos.py https://github.com/google-research/bert.git ../repos/bert ./bert.txt eedf5716ce1268e56f0a50264a88cafad334ac61
python crawl_repos.py https://github.com/dmlc/dgl.git ../repos/dgl ./dgl.txt f5c8c0b500db810210d22add9b7d6fc809edc0d2
python crawl_repos.py https://github.com/facebookresearch/esm.git ../repos/esm ./esm.txt 2b369911bb5b4b0dda914521b9475cad1656b2ac
python crawl_repos.py https://github.com/xmu-xiaoma666/External-Attention-pytorch.git ../repos/External-Attention-pytorch ./External-Attention-pytorch.txt 90e774ed4a2a734fe91dde6a4a737de0f0c54122
python crawl_repos.py https://github.com/IDEA-Research/Grounded-Segment-Anything.git ../repos/Grounded-Segment-Anything ./Grounded-Segment-Anything.txt d1c510118c0fdffb87bab3cd3dfe3e6b02a36f8b
python crawl_repos.py https://github.com/deep-floyd/IF.git ../repos/if ./if.txt ffc8163891682beaca0c5eb6b9077860a3bc6509
python crawl_repos.py https://github.com/salesforce/LAVIS.git ../repos/lavis ./lavis.txt 7f00a08
python crawl_repos.py https://github.com/vinits5/learning3d.git ../repos/learning3d ./learning3d.txt becc4a0017488390d3f04e4bc2ce9efb368115c5
python crawl_repos.py https://github.com/microsoft/muzic.git ../repos/muzic ./muzic.txt 9d52ff30b209a19fe51606242797c51a1bf7d7c1
python crawl_repos.py https://github.com/mlfoundations/open_clip.git ../repos/open_clip ./open_clip.txt 91923dfc376afb9d44577a0c9bd0930389349438
python crawl_repos.py https://github.com/eriklindernoren/PyTorch-GAN.git ../repos/PyTorch-GAN ./PyTorch-GAN.txt 36d3c77e5ff20ebe0aeefd322326a134a279b93e
python crawl_repos.py https://github.com/huggingface/pytorch-image-models.git ../repos/pytorch-image-models ./pytorch-image-models.txt 205d8ad37c838c20c541b787dca840edd9ed1408
python crawl_repos.py https://github.com/thuml/Time-Series-Library.git ../repos/Time-Series-Library ./Time-Series-Library.txt 51e80b6f76b7823fea2512d6d170b1b83b149d2d
python crawl_repos.py https://github.com/NVIDIA/vid2vid.git ../repos/vid2vid ./vid2vid.txt 2e6d13755fc2e33200e7d4c0c44f2692d6ab0898


python merge_test_set.py --max_workers 10 --split full

python merge_test_set.py --max_workers 10 --split quarter
