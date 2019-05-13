from torch import nn
import torch
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
import os
import cv2
import pandas as pd
import numpy as np
import cv2
from ResNet import ResNet
from my_data import Cat_Dataset
#
def read_img(dir):
    imgs = []
    files = os.listdir(train_dir)
    for file in files:
        file_path = train_dir + '/' + file
        img = cv2.imread(file_path)
        imgs.append(img)
    return imgs

train_dir = 'E:/keras_cat/train'
test_dir = 'E:/keras_cat/test'

X_train = read_img(train_dir)
X_test = read_img(test_dir)
# print(files)
# 用来显示图片
# im2 = cv2.resize(imgs[0], (256,256), interpolation=cv2.INTER_CUBIC)
# cv2.imshow('image', im2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def get_lab(dir):
    labels = []
    files = os.listdir(train_dir)
    for file in files:
        file_lable = file.split('_', 1)[0]
        labels.append(file_lable)
    return labels
y_train = get_lab(train_dir)
y_test = get_lab(test_dir)


def train(model, device, train_loader, optimizer, epochs, criterion):
    total_step = len(train_loader)
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 4 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, epochs, i + 1, total_step, loss.item()))


def val(model, device, test_loader):
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the 42 test images: {} %'.format(100 * correct / total))

    torch.save(model.state_dict(), './checkpoint/cat_first.pth')

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyper parameters
    num_epochs = 80
    num_classes = 10
    batch_size = 100
    learning_rate = 0.01

    # transform = transforms.Compose([
    #     transforms.Pad(4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomCrop(32),
    #     transforms.ToTensor()])


    train_dataset = Cat_Dataset(X_train, y_train,
                       transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                       ]))

    test_dataset = Cat_Dataset(X_test, y_test,
                       transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                       ]))

    # Image preprocessing modules




    # CIFAR-10 dataset
    # train_dataset = torchvision.datasets.CIFAR10(root='../../data/',
    #                                              train=True,
    #                                              transform=transform,
    #                                              download=True)
    #
    # test_dataset = torchvision.datasets.CIFAR10(root='../../data/',
    #                                             train=False,
    #                                             transform=transforms.ToTensor())
    #
    # # Data loader
    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
    #                                            batch_size=batch_size,
    #                                            shuffle=True)
    #
    # test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
    #                                           batch_size=batch_size,
    #                                           shuffle=False)
    #
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = ResNet(num_classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train( model, device, train_loader, optimizer=optimizer, epochs=num_epochs , criterion = criterion)
    val(model, device, test_loader)

if __name__ == '__main__':
    main()