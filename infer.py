from PIL import Image 
from transformers import SamProcessor
from torchvision import transforms
import os 
import numpy as np
from transformers import SamModel 
from torch.optim import Adam
import monai
from tqdm import tqdm
from statistics import mean
import torch
import glob 
from dataloading import build_loader
import wandb
from PIL import Image 

T2=transforms.ToPILImage()

# args

image_path="loco2/images/subset1_1.jpg"
resume_point="out_1"
vitbase=True
vitlarge=False
cuda=True
save_name="outputs3"



# processor
if vitbase:
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
elif vitlarge:
    processor = SamProcessor.from_pretrained("facebook/sam-vit-large")
else:
    raise Exception("no processor")

if cuda:
    device = "cuda"
else:
    device="cpu"

image=Image.open(image_path)
inputs = processor(image, input_boxes=None, return_tensors="pt").to(device)

# model 
model = SamModel.from_pretrained(resume_point)
model.eval()
model=model.to(device)

# Inference 

with torch.no_grad():
  outputs = model(**inputs, multimask_output=True)



outputs=outputs.pred_masks.squeeze(1)
outputs=outputs.squeeze(0)
outputs=T2(outputs)
outputs.save(f"{save_name}.png")
print("Save Complete")