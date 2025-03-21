import pickle
import numpy as np
from PIL import Image
from data.base_dataset import BaseDataset, get_transform
import os

class PatchDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.transform = get_transform(opt, grayscale=(input_nc == 1))
        
        # 加载数据集
        self.raw_data = pickle.load(open(opt.dataroot, "rb"))
        print(f"加载了 {len(self.raw_data)} 对图像块")

    def __getitem__(self, index):
        # 获取图像对
        data_pair = self.raw_data[index]
        
        # 读取A域图像
        A_path = data_pair['A']
        A_img = Image.open(A_path).convert('L')  # 转换为灰度图
        
        # 读取B域图像
        B_path = data_pair['B']
        B_img = Image.open(B_path).convert('L')  # 转换为灰度图
        
        # 应用变换
        if self.transform:
            A_img = self.transform(A_img)
            B_img = self.transform(B_img)
        
        return {'A': A_img, 'B': B_img}

    def __len__(self):
        return len(self.raw_data) 