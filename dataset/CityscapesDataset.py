import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import albumentations as A

class CityscapesDataset(Dataset):
    def __init__(self, split, root_dir, target_type='semantic', mode='fine', transform=None, eval=False):
        self.transform = transform
        if mode == 'fine':
            self.mode = 'gtFine'
        
        elif mode == 'coarse':
            self.mode = 'gtCoarse'
        
        self.split = split
        self.yLabel_list = []
        self.XImg_list = []
        self.eval = eval

        self.label_path = os.path.join(os.getcwd(), root_dir+'/'+self.mode+'/'+self.split)
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
            # image = transforms.ToTensor()(image)
            # image = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(image)
            transformed=self.transform(image=np.array(image), mask=np.array(y))
            image = transformed["image"]
            # transformed=self.transform[:-1](mask=np.array(y))
            y = transformed["mask"]
        # image = transforms.ToTensor()(image)
        # y = np.array(y)
        # y = torch.from_numpy(y)
        
        y = y.type(torch.LongTensor)
    
        return image, y, self.XImg_list[index], self.yLabel_list[index]
