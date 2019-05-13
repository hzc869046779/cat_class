from torch import nn
import torch
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
import os
import cv2
import pandas as pd
import numpy as np

num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.01

mnist_train = pd.read_csv('./data/train.csv')
mnist_test = pd.read_csv('./data/test.csv')
# print(mnist_train.describe())

X_train = mnist_train[:,]

# print(len(train_dataset))
# print(train_dataset[0][0])

# a = train_dataset[0][0]
# b = a.numpy()
# b = b * 255
# # b = np.squeeze(b)
#
# b = b.reshape(28,28)
#
# print(np.shape(b))

# test_image = train_dataset[0][0].np()
# print(test_image)


# 显示图片
# im2 = cv2.resize(b, (256,256), interpolation=cv2.INTER_CUBIC)
# cv2.imshow('image', im2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
