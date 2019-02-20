# -*- coding:utf-8 -*-
# author:zhangwei

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils import data
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

'''
   加载数据
'''
traincsv = pd.read_csv("G:\\diabetes\\data\\train.csv", dtype=np.float32)
targets_numpy = traincsv.label.values
features_numpy = traincsv.loc[:,traincsv.columns != "label"].values / 255                # normalizing the data
features_train, features_test, targets_train, targets_test = train_test_split(features_numpy,
                                                                              targets_numpy,
                                                                              test_size=0.2,
                                                                              random_state=42)

featuresTrain = torch.from_numpy(features_train)
targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor) # data type is long

# create feature and targets tensor for test set.
featuresTest = torch.from_numpy(features_test)
targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor)

batch_size = 100
n_iters = 100000
num_epochs = n_iters / (len(features_train) / batch_size)
num_epochs = int(num_epochs)

train = data.TensorDataset(featuresTrain, targetsTrain)
test = data.TensorDataset(featuresTest, targetsTest)

# data loader
train_loader = data.DataLoader(train,
                               batch_size=batch_size,
                               shuffle=False)
test_loader = data.DataLoader(test,
                              batch_size=batch_size,
                              shuffle=False)

class ANNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ANNModel, self).__init__()

        # linear function1  784 inputs ->100 outputs
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # non-linear function 1
        self.relu1 = nn.ReLU()

        # linear fucntion2 100->100
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # non-linear function 2
        self.tanh2 = nn.Tanh()

        # linear function3 100->100
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        # non-linear function 3
        self.elu3 = nn.ELU()

        # linear fucntion 4 for output 100->10
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # linear function 1
        out = self.fc1(x)
        # Non-linearity 1
        out = self.relu1(out)

        # Linear function 2
        out = self.fc2(out)
        # Non-linearity 2
        out = self.tanh2(out)

        # Linear function 2
        out = self.fc3(out)
        # Non-linearity 2
        out = self.elu3(out)

        # Linear function 4 (readout)
        out = self.fc4(out)
        return out

input_dim = 28*28
hidden_dim = 100 #hidden layer dim is one of the hyper parameter and it should be chosen and tuned. For now I only say 150 there is no reason.
output_dim = 10

# create ANN
model = ANNModel(input_dim, hidden_dim, output_dim)

# cross entropy loss
error = nn.CrossEntropyLoss()

# using SGD for optimizing the loss and updating the parameters
learning_rate = 0.02
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# ANN traing with training data set
count = 0
loss_list = []
iteration_list = []
accuracy_list = []

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        train = Variable(images.view(-1, 28 * 28))
        labels = Variable(labels)

        # Clear gradients
        optimizer.zero_grad()

        # Forward propagation
        outputs = model(train)

        # Calculate softmax and ross entropy loss
        loss = error(outputs, labels)

        # Calculating gradients
        loss.backward()

        # Update parameters
        optimizer.step()

        count += 1

        if count % 50 == 0:
            # Calculate Accuracy
            correct = 0
            total = 0
            # Predict test dataset
            for images, labels in test_loader:
                test = Variable(images.view(-1, 28 * 28))

                # Forward propagation
                outputs = model(test)

                # Get predictions from the maximum value
                predicted = torch.max(outputs.data, 1)[1]

                # Total number of labels
                total += len(labels)

                # Total correct predictions
                correct += (predicted == labels).sum()

            accuracy = 100 * correct / float(total)

            # store loss and iteration
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
            if count % 500 == 0:
                # Print Loss
                print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data, accuracy))
