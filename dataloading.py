from torch.utils import data
from torchvision import transforms
from datasets import Dataset
from PIL import Image 
from torch.utils.data import DataLoader
import numpy as np

T1=transforms.ToTensor()
T2=transforms.ToPILImage()


def build_loader(img_dir,label_dir,processor,batch):
     
    my_list=[]
    for i in range(len(img_dir)):
        dict={}
        dict["image"]=Image.open(img_dir[i])
        dict["label"]=Image.open(label_dir[i])
        my_list.append(dict)


    dataset = Dataset.from_list(my_list)
    dataset2=SAMDataset(dataset,processor)
    dataloader = DataLoader(dataset2, batch_size=batch, shuffle=True)
    return dataloader






class SAMDataset(data.Dataset):
  def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

  def __len__(self):
        return len(self.dataset)

  def __getitem__(self, idx):

        item = self.dataset[idx]
        image = item["image"]
        ground_truth_mask = np.array(item["label"].resize((256,256)))

        inputs = self.processor(image, input_boxes=None, return_tensors="pt")
        inputs = {k:v.squeeze(0) for k,v in inputs.items()}    
        inputs["ground_truth_mask"] = T1(ground_truth_mask)

        return inputs