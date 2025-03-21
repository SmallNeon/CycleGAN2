import os
from data.base_dataset import BaseDataset, get_transform
import SimpleITK as sitk
import random
import numpy as np
import pickle


class UnalignedCtDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A.pickle')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B.pickle')  # create a path '/path/to/data/trainB'

        self.A_paths = pickle.load(open(self.dir_A, "rb"))   # load images from '/path/to/data/trainA'
        self.B_paths = pickle.load(open(self.dir_B, "rb"))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        # btoA = self.opt.direction == 'BtoA'
        # input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        # output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        # self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        # self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        A_image = sitk.ReadImage(A_path)
        A_image = sitk.GetArrayFromImage(A_image)
        A_image = A_image / 1e3
        A_image = np.expand_dims(A_image, 0).astype(np.float64)       

        B_image = sitk.ReadImage(B_path)
        B_image = sitk.GetArrayFromImage(B_image)
        # B_image[B_image < 0] = 0
        # B_image = resize(B_image,(256, 256),anti_aliasing=True)
        # B_image = B_image / B_image.max()
        # B_image = B_image * 2 - 1
        B_image = B_image / 1e3
        # B_image = B_image - 1
        B_image = np.expand_dims(B_image, 0).astype(np.float64)


        return {'A': A_image, 'B': B_image, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
