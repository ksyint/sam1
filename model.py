import torch 
from transformers import SamProcessor
from transformers import SamModel
import torch.nn as nn 


def build_model(cfg):

    if cfg.checkpoint is not None:
        model = SamModel.from_pretrained(cfg.checkpoint)

    else:
        model=SamModel.from_pretrained(cfg.backbone)

    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            if cfg.freeze==True:
                param.requires_grad_(False) 
            else:
                param.requires_grad_(True) 


    if cfg.multi_gpu==False:
        model=model.to(f"cuda:{cfg.device}")  

    else:
        model=nn.DataParallel(model)
        model=model.cuda()

    return model 