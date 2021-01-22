import numpy as np
import torch
from torch.utils.data import Dataset
import os
#from skimage import io, transform
from utils.mypath import MyPath
#from torchvision import transforms, utils
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
import cv2
import PIL
from PIL import Image


class Hep2(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir="/content/small",train=True, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir=root_dir
        self.root_txt = root_dir+"\\Train.txt" if train else root_dir+"\\Test.txt"
        f=open(self.root_txt,"r")
        self.img_list = f.readlines()
        self.transform = transform
        self.classes=['THP1','MCF7','PBMC']

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = cv2.imread(self.root_dir+"\\"+self.img_list[idx].split(" ")[0])
        img_size = (image.shape[0],image.shape[1])
        image = Image.fromarray(image)
        #print("size of the data item",image.shape)
        target = int(self.img_list[idx].split(" ")[1])
        class_name = self.classes[target]

        if self.transform:
            image = self.transform(image)

        sample = {'image': image,'target': target,'meta': {'im_size': img_size, 'index': idx, 'class_name': class_name}}
        return sample
    
    def get_image(self,index):
        image = cv2.imread(self.root_dir+"\\"+self.img_list[idx].split(" ")[0])
        return image
