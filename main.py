import glob 
from PIL import Image 
from datasets import Dataset
import numpy as np
from transformers import SamProcessor
from transformers import SamModel
import monai
from torch.utils.data import DataLoader
from tqdm import tqdm
import os 
from statistics import mean
import wandb
import torch.nn as nn
import config as cfg
from model import build_model
from utils import build_optimizer
import torch
from data.dataloader import SAMDataset,build_data_list


wandb.init(project=cfg.wandb_project)
wandb.run.name = cfg.wandb_run_name
wandb.run.save()


data_list=build_data_list(cfg)
processor = SamProcessor.from_pretrained(cfg.backbone)
train_dataset = SAMDataset(data_list,processor)
train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)


model=build_model(cfg)
optimizer=build_optimizer(cfg,model)
criterion = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')


if cfg.multi_gpu==True:
    
    device=torch.device("cuda")

else:
    
    device=torch.device(f"cuda:{cfg.device}")


model.train()
for epoch in range(cfg.epochs):
    epoch_losses = []
    
    for batch in tqdm(train_dataloader):
        
      
      outputs = model(pixel_values=batch["pixel_values"].to(device),
                      input_boxes=None,
                      multimask_output=True)
     
    

      predicted_masks = outputs.pred_masks.squeeze(1)
     

      ground_truth_masks = batch["ground_truth_mask"].float().to(device)
      ground_truth_masks=ground_truth_masks.permute(0,3,1,2)
      
      loss = criterion(predicted_masks, ground_truth_masks)
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      epoch_losses.append(loss.item())
      wandb.log({"Train Batch Loss": loss.item()})
      


    print(f'EPOCH: {epoch+1}')
    print(f'Mean loss: {mean(epoch_losses)}')
    wandb.log({"Train Loss": mean(epoch_losses)})
    
    os.makedirs(cfg.save_dir,exist_ok=True)
    model.save_pretrained(cfg.save_dir)
    print("Model Saved")
