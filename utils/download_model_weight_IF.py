from deepfloyd_if.modules import IFStageI, IFStageII, StableStageIII
from deepfloyd_if.modules.t5 import T5Embedder
from deepfloyd_if.pipelines import dream
from huggingface_hub import login

login()

device = 'cuda'
if_I = IFStageI('IF-I-L-v1.0', device=device)
if_I = IFStageI('IF-I-XL-v1.0', device=device)
if_II = IFStageII('IF-II-M-v1.0', device=device)
if_III = StableStageIII('stable-diffusion-x4-upscaler', device=device)
if_I = IFStageI('IF-I-M-v1.0', device=device)
if_II = IFStageII('IF-II-L-v1.0', device=device)
t5 = T5Embedder(device='cpu')



