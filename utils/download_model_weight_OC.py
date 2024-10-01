import os
import numpy as np
import torch
import open_clip


open_clip.list_pretrained()
open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
open_clip.create_model_and_transforms('coca_ViT-B-32', pretrained='mscoco_finetuned_laion2b-s13b-b90k')
open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion400m_e31')
open_clip.create_model_and_transforms('convnext_base', pretrained='laion400m_s13b_b51k')
open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion400m_e31')
open_clip.create_model_and_transforms('ViT-B-32', pretrained='datacomp_m_s128m_b4k')
open_clip.create_model_and_transforms('RN50', pretrained='openai')
open_clip.create_model_and_transforms('ViT-L-14', pretrained='datacomp_xl_s13b_b90k')
open_clip.create_model_and_transforms('convnext_large_d', pretrained='laion2b_s26b_b102k_augreg')
open_clip.create_model_and_transforms('ViT-B-32', pretrained='datacomp_m_s128m_b4k')
open_clip.create_model_and_transforms('RN50', pretrained='openai')
open_clip.create_model_and_transforms('ViT-L-14', pretrained='datacomp_xl_s13b_b90k')
open_clip.create_model_and_transforms('convnext_large_d', pretrained='laion2b_s26b_b102k_augreg')
open_clip.create_model_and_transforms('ViT-B-32', pretrained='commonpool_m_basic_s128m_b4k')
open_clip.create_model_and_transforms('convnext_xxlarge', pretrained='laion2b_s34b_b82k_augreg_soup')
open_clip.create_model_and_transforms('ViT-L-14', pretrained='commonpool_xl_clip_s13b_b90k')
open_clip.create_model_and_transforms('EVA02-E-14', pretrained='laion2b_s4b_b115k')
open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')

