# -*- coding: utf-8 -*-
"""
==============================================================================
Time : 2022/9/4 11:22
File : 2_simple_fnn.py

全连接网络
视频链接：https://www.youtube.com/watch?v=Jy4wM2X21u0&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=3&ab_channel=AladdinPersson

==============================================================================
"""

import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.


# 构建网络
class NN(nn.Module):
    def __init__(self,  input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x =self.fc2(x) # 后面是交叉熵 ，可以不用 softmax
        return x


# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数
input_size = 784
num_classes = 10
lr = 0.001
batch_size= 64
num_epochs = 1

# load data
train_dataset = datasets.MNIST(root='../../corpus/', train=True, transform=transforms.ToTensor(),
                               download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='../../corpus/', train=False, transform=transforms.ToTensor(),
                               download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# initialize network
model = NN(input_size=input_size, num_classes=num_classes).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)


# Check accuracy on training to see how good our model is
def check_accuracy(loader, model):
    if loader.dataset.train:
        print('checking accuracy on training data')
    else:
        print('checking accuracy on test data')

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}"
        )

    model.train()



if __name__ == "__main__":
    pass
    test_model = NN(784, 10)
    x =torch.randn(64, 784)
    print(test_model(x).shape)
    # ---------------------------
    # train networks
    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(train_loader):
            # get data to CUDA  if possible
            data = data.to(device=device)  # ([64, 1, 28, 28])
            targets = targets.to(device=device)

            # get to correct shape
            data = data.reshape(data.shape[0], -1) # [64, 784]

            # forward
            scores = model(data)
            loss = criterion(scores, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradent descend or adam step
            optimizer.step()

    check_accuracy(train_loader, model)
    check_accuracy(test_loader, model)



