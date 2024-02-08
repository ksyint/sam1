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


#args

batch=12
num_epochs = 600
save_every_epoch=20
resume=False

resume_point="out_1"
vitbase=True
vitlarge=False
cuda=True

#wandb 

wandb.init(project="sam")
wandb.run.name = '2'
wandb.run.save()


#data loading 

img_dir=glob.glob("/root/sam2/loco2/images/*.jpg")[0:800]
label_dir=glob.glob("/root/sam2/loco2/masks/*.jpg")[0:800]

img_dir2=glob.glob("/root/sam2/loco2/images/*.jpg")[800:]
label_dir2=glob.glob("/root/sam2/loco2/masks/*.jpg")[800:]

if vitbase:
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
elif vitlarge:
    processor = SamProcessor.from_pretrained("facebook/sam-vit-large")
else:
    raise Exception("no processor")


train_loader=build_loader(img_dir,label_dir,processor,batch)
valid_loader=build_loader(img_dir2,label_dir2,processor,batch)


#model loading 

if resume:
  
        model = SamModel.from_pretrained(resume_point)
else:

    if vitbase:

        model = SamModel.from_pretrained("facebook/sam-vit-base")
    elif vitlarge:

        model = SamModel.from_pretrained("facebook/sam-vit-large")
    else:
       
       raise Exception("only one vit")
       
for name, param in model.named_parameters():
  if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
    param.requires_grad_(False)

#train pipeline
    

optimizer = Adam(model.mask_decoder.parameters(), lr=2e-4, weight_decay=0)
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
if cuda:
    device = "cuda"
else:
   device="cpu"
model.to(device)
model.train()


for epoch in range(num_epochs):
    epoch_losses = []
    for batch in tqdm(train_loader):
      
      outputs = model(pixel_values=batch["pixel_values"].to(device),multimask_output=True)
      predicted_masks = outputs.pred_masks.squeeze(1)
      ground_truth_masks = batch["ground_truth_mask"].float().to(device)
      loss = seg_loss(predicted_masks, ground_truth_masks)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      epoch_losses.append(loss.item())
      

    print(f'EPOCH: {epoch+1}')
    print(f'Train Mean loss: {mean(epoch_losses)}')
    wandb.log({"Train Loss":mean(epoch_losses)})

    if epoch%save_every_epoch==0:
      os.makedirs(f"checkpoint/sam_out_{epoch+1}",exist_ok=True)
      model.save_pretrained(f"checkpoint/sam_out_{epoch+1}")

    model.eval()
    epoch_losses = []
    with torch.no_grad():
        for batch in tqdm(valid_loader):

            outputs = model(pixel_values=batch["pixel_values"].to(device),multimask_output=True) 
            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)
            loss = seg_loss(predicted_masks, ground_truth_masks)

            epoch_losses.append(loss.item())


        print(f'EPOCH: {epoch+1}')
        print(f'Valid Mean loss: {mean(epoch_losses)}')
        wandb.log({"Valid Loss":mean(epoch_losses)})
        model.train()

