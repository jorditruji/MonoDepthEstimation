from __future__ import division
import torch
from torch.utils import data
import numpy as np
import time
from PIL import Image
import cv2
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose
)
from albumentations.pytorch import ToTensor
import time

'RECORDATORI SCANNET dataset: \
Max depth(uint16): 9998 \
Min depth(uint16) : 264 \
Mean RGB: [0.4944742  0.4425867  0.38153833] \
Std RGB: [0.23055981 0.22284868 0.21425385] '

'RECORDATORI nyu_V2 dataset: \
[0.48958883 0.41837043 0.39797969] \
[0.26429949 0.2728771  0.28336788]'

def read_image(file):
    '''
    Read a RGB mage and returns it as a PIL RGB image

    Args:
        file (int): Path to image

    Returns:
        rgb_frame (PIL): RGB color image


    '''
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(file, 'rb') as f:
        img = Image.open(f).convert('RGB')

    return img



class GenericDataset(data.Dataset):

    """
    Reads and augments the dataset images according to its tate and parameters. Base generic class implementing the common methods.

    :ivar depth_names (list): List of the depth images paths 
    :ivar is_train (boolean): Load images for train/inference 
    :ivar transforms (albumentation or str): Loads augmentator config from path if str and sets it to attr transforms
    """
    def __init__(self, depth_names, transforms = None):
        # Paths to dataset samples
        #self.is_train = is_train 
        self.depth_frames = depth_names 
        self.transforms = transforms
        self.RGB_frames = self._depth2RGB()

    def __len__(self):
        """
        Returns the amount of images of the dataset.


        Returns:
            dataset_len (int): Amount of images of the dataset


        """
        return len(self.depth_frames)

    def __getitem__(self, index):
        """
        Reads a complete sample of the dataset including its ground truth.

        Args:
            index (int): Index of the sample to retetrieve

        Returns:
            rgb_frame (Tensor): Image of shape (#channels, H, W)
            depth_frame (Tensor): Depth map of shape (1, H, W)
            filename (str): 

        """
        pass


        #return depth, rgb, self.depth_frames[index]



    def _depth2RGB(self):
        """
        Abstract method. It converts the path to depth_frames to match the color image paths.

        Returns:
            rgb_frames (list): List of image paths matching the color samples.
        """       
        pass



class NYUDataset(GenericDataset):
    """
    Reads and augments the dataset images according to its tate and parameters. NYU dataloader implementation
    Attributes:
        depth_names (list): List of the depth images paths. 
        is_train (boolean): State of the loader. Possible states are training/inference.
    """


    def _depth2RGB(self):
        '''Edit strings to match rgb image paths'''
        return [depth.replace('depth.png','rgb.png') for depth in self.depth_frames] 

    def read_depth(self, file):
        image = cv2.imread(file, -cv2.IMREAD_ANYDEPTH)

        image = np.asarray(image, dtype = float)
        image = (image-np.min(image))/(np.max(image)-np.min(image))

        return image        



    def __getitem__(self, index):
        """
        Reads a complete sample of the dataset including its ground truth.

        Args:
            index (int): Index of the sample to retetrieve

        Returns:
            rgb_frame (Tensor): Image of shape (#channels, H, W)
            depth_frame (Tensor): Depth map of shape (1, H, W)
            filename (str): 

        """
        start_time = time.time()
        depth = self.read_depth(self.depth_frames[index])
        rgb = np.asarray(read_image(self.RGB_frames[index]))

        sample =  {"image": rgb, "mask": depth}
        #print("Augmenting", depth)   
        augmented = self.transforms(**sample)
        rgb, depth  = augmented['image'], augmented['mask'] 
        #print("augmented", depth)
        print("{} seconds for reading sample.".format(time.time()-start_time))
        return depth, rgb, self.depth_frames[index]




if __name__ == '__main__':
    # Testing:
    # Sample data
    import matplotlib.pyplot as plt


    augm = strong_aug(0.9)

    depths = ['../sample_images/classroom_000310depth.png','../sample_images/classroom_000350depth.png',
              '../sample_images/classroom_000329depth.png']
    dataset = NYUDataset(depths, is_train = False, transforms=  augm)

    print(dataset.RGB_frames)
    img = read_image(dataset.RGB_frames[-1])
    plt.imshow(img)
    plt.figure()
    depth = dataset.read_depth(dataset.depth_frames[-1])
    print(depth.dtype)
    print(np.max(depth))
    print(np.min(depth))

    plt.imshow(depth, cmap='Greys_r')

    data = {"image": np.array(img), "mask": depth}
    augm = strong_aug(0.9)
    A.save(augm, 'transform_prova.json')

    fig=plt.figure()
    fig=plt.figure()
    labels = []
    columns = 2
    rows = 2
    for i in range(1, columns*rows +1):
        augmented = augm(**data)
        fig.add_subplot(rows, columns, i)
        plt.imshow( augmented["mask"],  cmap='Greys_r')
        labels.append( augmented["image"])

    fig=plt.figure()

    for _i, i in enumerate(range(1, columns*rows +1)):
        fig.add_subplot(rows, columns, i)
        plt.imshow( labels[_i])

    plt.show()
    print(dataset.transforms)

