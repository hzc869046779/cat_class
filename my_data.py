import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from imgaug import augmenters as iaa
import imgaug
import os
import scipy.ndimage
import scipy.misc
import random
from PIL import Image


class Cat_Dataset(Dataset):

    def __init__(self, samples_npy, annotations_npy, transform):
        super().__init__()
        self.samples = samples_npy
        self.annotations = annotations_npy
        # self.input_size = input_size
        self.transform = transform
        # self.mode = mode

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        img = self.transform(self.samples[idx])
        label = int(self.annotations[idx])

        return img, label
