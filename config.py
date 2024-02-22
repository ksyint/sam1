import torch 

wandb_project="sam 0222_2"
wandb_run_name="2"
save_dir="checkpoint2"

batch_size=14
epochs=1000
learning_rate=1e-5
weight_decay=1e-6
optimizer="adam"

backbone="facebook/sam-vit-base"
checkpoint="checkpoint"
freeze=True #vison_encoder, prompt_encoder 

multi_gpu=False
device=0

img_dir="samples/input"
mask_dir="samples/mask"
img_size=256

