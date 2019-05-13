from torch import nn
import torch
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms

# Device configuration
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#
# # Hyper parameters
# num_epochs = 5
# num_classes = 10
# batch_size = 100
# learning_rate = 0.1
#
#
# # Device configuration
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#
# # Hyper parameters
# num_epochs = 5
# num_classes = 10
# batch_size = 100
# learning_rate = 0.01
#
# # MNIST dataset
# train_dataset = torchvision.datasets.MNIST(root='../../data/',
#                                            train=True,
#                                            transform=transforms.ToTensor(),
#                                            download=True)
#
# test_dataset = torchvision.datasets.MNIST(root='../../data/',
#                                           train=False,
#                                           transform=transforms.ToTensor())
#
# # Data loader
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                            batch_size=batch_size,
#                                            shuffle=True)
#
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                           batch_size=batch_size,
#                                           shuffle=False)
# print(len(train_dataset))
# print(len(test_dataset))

class ResidualBlock(nn.Module):

    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel)
        )

        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()

        self.pre = nn.Sequential(
            nn.Conv2d(3, 32, 7, 2, 3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)

        )

        self.layer1 = self._make_layer(32, 48, 3)
        self.layer2 = self._make_layer(48, 64, 4, stride=2)
        self.layer3 = self._make_layer(64, 128, 6, stride=2)
        self.layer4 = self._make_layer(128, 256, 3, stride=2)

        self.fc = nn.Linear(256 , num_classes)

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):

        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel)
        )

        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# model = ResNet(num_classes).to(device)
#
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
#
#
#
#
# # Train the model
# total_step = len(train_loader)
# for epoch in range(num_epochs):
#     for i, (images, labels) in enumerate(train_loader):
#         images = images.to(device)
#         labels = labels.to(device)
#
#         # Forward pass
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#
#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         if (i + 1) % 100 == 0:
#             print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
#                   .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
#
# # Test the model
# model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in test_loader:
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
#     print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
#
# # Save the model checkpoint
# torch.save(model.state_dict(), './checkpoint/first.pth')



