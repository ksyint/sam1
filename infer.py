from transformers import SamProcessor
from transformers import SamModel
import torch 
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image 
import cv2
import numpy as np
import glob 
import os 

pallet_list=[(255,0,0),(0,255,0),(0,0,255)]
class_list=["forklift","pallet","pallet_truck"]

inputs=glob.glob("samples/input/*.jpg")
for input_path in inputs:

    name=input_path.split("/")[-1]
    device = "cuda:0" 

    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    model = SamModel.from_pretrained("/root/sam4/sam3/code/vit_base_sam")
    model.to(device)

    image=Image.open(input_path)
    image=image.resize((256,256))
    inputs = processor(image, input_boxes=None, return_tensors="pt").to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=True)

    medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
    medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
    medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)

    image=np.array(image)
    for i in range(3):

        mask=np.expand_dims(medsam_seg[:,:,i],axis=-1)
        mask2=np.expand_dims(medsam_seg[:,:,i],axis=-1)
        mask3=np.expand_dims(medsam_seg[:,:,i],axis=-1)
        new_mask = np.stack([mask, mask2, mask3], axis=-1)
        new_mask=new_mask.squeeze(2)

        cyan = np.full_like(image,pallet_list[i])
        blend = 0.8
        img_cyan = cv2.addWeighted(image, blend, cyan, 1-blend, 0)
        result = np.where(new_mask>0.0, img_cyan, image)

        result=Image.fromarray(result)
        result=result.resize((512,512))
        os.makedirs("output2")
        result.save(f"output2/mask_of_{class_list[i]}_{name}")
