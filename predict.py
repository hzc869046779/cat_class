from torch import nn
import torch
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import matplotlib.image as mpimg
import matplotlib as plt
from ResNet import ResNet
import os
import cv2

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

model = ResNet(4)
checkpoint = torch.load('./checkpoint/cat_first.pth')
model.load_state_dict(checkpoint)
model.eval()

test_dir = './data/test01.jpg'
img = cv2.imread(test_dir)
img = transform(img)
_, predicted = torch.max(img.data, 1)


print(predicted)


