#Cityscapes dataset is used in this project
import os
from PIL import Image 
import torch
from torch.utils.data import Dataset
import numpy as np


class CityscapesDataset(Dataset):
    def __init__(self,root_dir, split, transform=None):
        self.transform = transform

        self.split = split

        self.label_path = os.path.join(os.getcwd(), root_dir+'/gtFine'+self.split)
        self.rgb_path = os.path.join(os.getcwd(), root_dir+'/leftImg8bit/'+self.split)
        city_list = sorted(os.listdir(self.label_path))

        for city in city_list:
            temp = os.listdir(self.label_path+'/'+city)
            list_items = temp.copy()
    
            # 19-class label items being filtered
            for item in temp:
                if not item.endswith('labelTrainIds.png', 0, len(item)):
                    list_items.remove(item)

            # defining paths
            list_items = ['/'+city+'/'+path for path in list_items]

            self.yLabel_list.extend(sorted(list_items))
            self.XImg_list.extend(
                ['/'+city+'/'+path for path in sorted(os.listdir(self.rgb_path+'/'+city))]
            )

    def __len__(self):
        length = len(self.XImg_list)
        return length

    def __getitem__(self, index):
        image = Image.open(self.rgb_path+self.XImg_list[index])
        y = Image.open(self.label_path+self.yLabel_list[index])

        if self.transform is not None:
            transformed=self.transform(image=np.array(image), mask=np.array(y))
            image = transformed["image"]
            y = transformed["mask"]
        
        y = y.type(torch.LongTensor)
    
        return image, y, self.XImg_list[index], self.yLabel_list[index]