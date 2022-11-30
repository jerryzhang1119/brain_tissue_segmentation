import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from args import ARGS
from datasets.data_augmentation import data_augmentation, _random_crop

class ChaseDB1Dataset(Dataset):
    def __init__(self, data_path, label_path, edge_path, need_enhance=True):
        super(ChaseDB1Dataset, self).__init__()
        self.data_path = data_path
        self.label_path = label_path

        
        self.need_enhance = need_enhance
        self.data_list = [_ for _ in os.listdir(data_path) if '.png' in _]
        self.label_list = [_ for _ in os.listdir(label_path) if '.png' in _]

        self.data_list = sorted(self.data_list)
        self.label_list = sorted(self.label_list)

        assert len(self.data_list) == len(self.label_list), \
            f"The number of data ({len(self.data_list)}) doesn't match the number of labels ({len(self.label_list)})"
        # Try to match data_list and label_list

    def __getitem__(self, index: int):
        img = Image.open(os.path.join(self.data_path, self.data_list[index]))
        label = Image.open(os.path.join(self.label_path, self.label_list[index]))


        # Data Augmentation
        # transform = transforms.Compose([
        #     transforms.RandomResizedCrop(scale=(0.8, 1.2)), # 0.5, 2
        # ])

            
        img, label = _random_crop(img, label)

        label = label.convert('L')


        img = np.array(img).transpose((2, 0, 1)) / 255. # [0, 255] -> [0, 1]
        mean = np.array([123.68, 116.779, 103.939]).reshape((3, 1, 1)) / 255.
        img = torch.FloatTensor(img - mean)

        label = torch.LongTensor(np.array(label)) # 512 x 512 # [0, 255] -> [0, 1]

        
        return {'image': img, 'label': label}
    
    def __len__(self) -> int:
        return len(self.data_list)


class ChaseDB1TestDataset(Dataset):
    def __init__(self, data_path):
        super(ChaseDB1TestDataset, self).__init__()
        self.data_path = data_path
        self.data_list = [_ for _ in os.listdir(data_path) if '.png' in _]

    def __getitem__(self, index: int):
        img = Image.open(os.path.join(self.data_path, self.data_list[index])) 
        img = np.array(img).transpose((2, 0, 1)) / 255.
        mean = np.array([123.68, 116.779, 103.939]).reshape((3, 1, 1)) / 255.
        img = torch.FloatTensor(img - mean)
        img_c, img_h, img_w = img.size()
        mask = torch.ones((img_h, img_w)).long()

        return {'image': img, 'mask': mask, 'filename': self.data_list[index]}
    
    def __len__(self) -> int:
        return len(self.data_list)

class ChaseDB1MetricDataset(ChaseDB1Dataset):
    def __init__(self, data_path, label_path, edge_path):
        super(ChaseDB1MetricDataset, self).__init__(data_path, label_path, edge_path, False)
        self.data_list = [_ for _ in os.listdir(data_path) if '.png' in _]
        self.label_list = [_ for _ in os.listdir(label_path) if '.png' in _]

        self.data_list = sorted(self.data_list)
        self.label_list = sorted(self.label_list)


    def __getitem__(self, index: int):
        img = Image.open(os.path.join(self.data_path, self.data_list[index]))
        label = Image.open(os.path.join(self.label_path, self.label_list[index]))


        label = label.convert('L')


        img = np.array(img).transpose((2, 0, 1)) / 255. # [0, 255] -> [0, 1]
        mean = np.array([123.68, 116.779, 103.939]).reshape((3, 1, 1)) / 255.
        img = torch.FloatTensor(img - mean)

        label = torch.LongTensor(np.array(label))  # 512 x 512 # [0, 255] -> [0, 1]


        img_c, img_h, img_w = img.size()
        mask = torch.ones((img_h, img_w)).long()
        return {'image': img, 'label': label, 'mask': mask,'filename': self.data_list[index]}
    
    def __len__(self) -> int:
        return len(self.data_list)
