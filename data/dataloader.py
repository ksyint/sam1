from torch.utils.data import Dataset
from PIL import Image 
import numpy as np 
import os 
from tqdm import tqdm
from datasets import Dataset as Dataset2

def build_data_list(cfg):
        
    img_list=os.listdir(cfg.img_dir)
    mask_list=os.listdir(cfg.mask_dir)

    assert len(img_list)==len(mask_list)

    main_list=[]
    
    for i, (img, mask) in enumerate(tqdm(zip(img_list, mask_list))):
            dict = {}
            img = Image.open(os.path.join(cfg.img_dir,img))
            mask = Image.open(os.path.join(cfg.mask_dir,mask))
            dict["image"] = img.resize((cfg.img_size, cfg.img_size))
            dict["label"] = mask.resize((cfg.img_size, cfg.img_size))
            main_list.append(dict)
            tqdm.write(f"{i + 1}/{len(img_list)}", end='\r')  
            

    data_list=Dataset2.from_list(main_list)


    return data_list
    



class SAMDataset(Dataset):
    
    
  def __init__(self, dataset, processor):
      
        self.dataset = dataset
        self.processor = processor

  def __len__(self):
      
        return len(self.dataset)

  def __getitem__(self, idx):
        
        item = self.dataset[idx]
        image = item["image"]
        ground_truth_mask = np.array(item["label"])
        inputs = self.processor(image, input_boxes=None, return_tensors="pt")
        inputs = {k:v.squeeze(0) for k,v in inputs.items()}
        inputs["ground_truth_mask"] = ground_truth_mask

        return inputs