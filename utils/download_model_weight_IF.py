from deepfloyd_if.modules import IFStageI, IFStageII, StableStageIII
from deepfloyd_if.modules.t5 import T5Embedder
from deepfloyd_if.pipelines import dream
from huggingface_hub import login
import os

if os.path.exists(os.path.expanduser('~/.huggingface/token')):
    print("Login token found. Skipping login.")
else:
    print("Login token not found. Attempting to login.")
    login()

device = 'cuda'
IFStageI('IF-I-L-v1.0', device=device)
IFStageI('IF-I-XL-v1.0', device=device)
IFStageII('IF-II-M-v1.0', device=device)
StableStageIII('stable-diffusion-x4-upscaler', device=device)
IFStageI('IF-I-M-v1.0', device=device)
IFStageII('IF-II-L-v1.0', device=device)
t5 = T5Embedder(device='cpu')



