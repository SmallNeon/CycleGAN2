# This code is released under the CC BY-SA 4.0 license.

import pickle

import numpy as np
import pydicom
from PIL import Image

from data.base_dataset import BaseDataset,get_transform
import os
import util
import matplotlib.pyplot as plt
from skimage.transform import resize
import SimpleITK as sitk

class CTDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.transform = get_transform(opt, grayscale=(input_nc == 1))           
        self.raw_data = pickle.load(open(opt.dataroot, "rb"))
        # self.raw_data = [os.path.join(opt.dataroot, path, "ARTERIAL", "DICOM") for path in os.listdir(opt.dataroot)]
        self.labels = None
        self.Aclass = opt.Aclass
        self.Bclass = opt.Bclass
        self._make_dataset()

    def _make_dataset(self):
        data = []     
        for entity in self.raw_data:
            if self.Aclass in entity:
                data.append(entity)
        self.raw_data = data

    def read_folder(self, folder_path):
        dicom_files = []
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            dicom = pydicom.dcmread(file_path)
            dicom_files.append(dicom)
        dicom_files.sort(key=lambda x: int(x.InstanceNumber))
    
        return np.array(list(map(lambda x: x.pixel_array, dicom_files)))

    def __getitem__(self, index):
        # Image from A
        # A_image = pydicom.dcmread(self.raw_data[index]).pixel_array
        A_image = sitk.ReadImage(self.raw_data[index])
        A_image = sitk.GetArrayFromImage(A_image)
        # A_image[A_image < 0] = 0
        # A_image = resize(A_image,(256, 256),anti_aliasing=True)
        A_image = A_image / 1e3
        # A_image = A_image - 1
        # A_image = A_image / A_image.max()
        # A_image = A_image * 2 - 1
        A_image = np.expand_dims(A_image, 0).astype(np.float64)       

        # Paired image from B
        # B_image = pydicom.dcmread(path).pixel_array
        B_image = sitk.ReadImage(self.raw_data[index].replace(self.Aclass, self.Bclass))
        B_image = sitk.GetArrayFromImage(B_image)
        # B_image[B_image < 0] = 0
        # B_image = resize(B_image,(256, 256),anti_aliasing=True)
        # B_image = B_image / B_image.max()
        # B_image = B_image * 2 - 1
        B_image = B_image / 1e3
        # B_image = B_image - 1
        B_image = np.expand_dims(B_image, 0).astype(np.float64)

        return {'A': A_image, 'B': B_image}

    def __len__(self):
        return len(self.raw_data)
