from models.Custom_model import *
import numpy as np
import pickle
import matplotlib.pyplot as plt
import dataLoader as DL
import torch

def ave_accuracy(output, y):
    return np.mean(np.argmax(output, axis=1).reshape(-1, 1) == y, axis=0)

def ave_accuracy_torch(output, y):
    return np.mean(np.argmax(output, axis=1) == y, axis=0)

def check_conv_forward():
    input_num = np.random.randint(0, 10, (10, 3, 10, 10))

    conv = Conv2d(in_channels=3, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), dilation=(0, 0),
                  padding_mode='zero')
    c = conv.forward(input_num)
    print((input_num[0, :, :3, :3] * conv.conv_w[:].T.reshape(1, 3, 3, 3)).sum() + conv.conv_b)
    print(c[0, 0, :1, :1])

def check_conv():
    conv = Conv2d(in_channels=2,out_channels=3)
    conv_w = np.transpose(conv.conv_w,axes=(1,0)).reshape(3,2,3,3)

    print(conv_w.shape)
    # exit(1)


    input_num = np.random.randint(0, 10, (1, 2, 5, 5))
    print(input_num)
    print(conv.im2col(input_num))
    # out = (input_num[0,:,:3,:3] * conv_w[0]).sum()+ conv.conv_b[0]
    # output = conv.forward(input_num)
    # print(input_num)
    # print('=' * 100)
    # print(output[0,0,0,0])
    # print(out)
    # print('=' * 100)
    # print(maxpool.argmax)

def check_maxpool():
    maxpool = maxpool2d()
    input_num = np.random.randint(0, 10, (1, 1, 4, 4))
    output = maxpool.forward(input_num)
    print(input_num)
    print('='*100)
    print(output)
    print('='*100)
    print(maxpool.argmax)


def train_model():
    # Use a breakpoint in the code line below to debug your script.
    batch_size = 20
    infile = open('mnist.pkl', 'rb')
    mnist_data = pickle.load(infile)
    infile.close()

    train_loader = DL.dataLoader(mnist_data['Xtrain'], mnist_data['ytrain'], batch_size)
    test_loader = DL.dataLoader(mnist_data['Xtest'], mnist_data['ytest'], batch_size)
    model = Custom_models(in_channels=1,layer_channels=[32,64],class_num=10,alpha=0.05)

    num_epochs = 100
    training_loss = {}
    training_accuracy  = {}

    num_train_batches = mnist_data['Xtrain'].shape[0] / batch_size
    criterion = crossentropy(class_num=10)
    ind = 0
    for e in range(1, num_epochs + 1):
        train_loader = DL.dataLoader(mnist_data['Xtrain'], mnist_data['ytrain'], batch_size)
        training_loss.update({e: 0.0})
        training_accuracy.update({e: 0.0})

        for image, label in train_loader:
            image = image.reshape(image.shape[0],1,20,20)
            # print(f'input shape = {image.shape}')
            pred = model.forward(image)

            loss = criterion.forward(z=pred, y=label)

            model.backward(criterion.backward())
            model.update()

            training_loss[e] += loss

            training_accuracy[e] += ave_accuracy(pred, label)

        # At each epoch 'e', print training loss and accuracy (homework)
        training_loss[e] = training_loss[e] / num_train_batches
        training_accuracy[e] = training_accuracy[e] / num_train_batches
        print(f'{e}th training loss = {training_loss[e]}\n{e}th training accuracy = {training_accuracy[e]}')

class models(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1,32,3,1,1)
        self.relu1 = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(2,2)

        self.conv2 = torch.nn.Conv2d(32,64,3,1,1)
        self.relu2 = torch.nn.ReLU()
        self.avg = torch.nn.AdaptiveAvgPool2d(1)

        self.fc = torch.nn.Linear(64,10)
        torch.nn.init.normal_(self.conv1.weight)

        torch.nn.init.normal_(self.conv2.weight)

        torch.nn.init.normal_(self.conv1.bias)

        torch.nn.init.normal_(self.fc.bias)

        torch.nn.init.normal_(self.fc.weight)

        torch.nn.init.normal_(self.conv1.bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.avg(x)
        x = torch.flatten(x,1)
        x = self.fc(x)

        return x

def train_model_torch():
    # Use a breakpoint in the code line below to debug your script.
    batch_size = 20
    infile = open('mnist.pkl', 'rb')
    mnist_data = pickle.load(infile)
    infile.close()

    train_loader = DL.dataLoader(mnist_data['Xtrain'], mnist_data['ytrain'], batch_size)
    test_loader = DL.dataLoader(mnist_data['Xtest'], mnist_data['ytest'], batch_size)
    model = Custom_models(in_channels=1,layer_channels=[32,64],class_num=10,alpha=0.05)

    num_epochs = 100
    training_loss = {}
    training_accuracy  = {}
    model = models()

    optim = torch.optim.SGD(model.parameters(),lr=0.05)
    num_train_batches = mnist_data['Xtrain'].shape[0] / batch_size
    loss_fn = torch.nn.CrossEntropyLoss()
    ind = 0
    for e in range(1, num_epochs + 1):
        train_loader = DL.dataLoader(mnist_data['Xtrain'], mnist_data['ytrain'], batch_size)
        training_loss.update({e: 0.0})
        training_accuracy.update({e: 0.0})

        for image, label in train_loader:
            image = image.reshape(image.shape[0],1,20,20)
            image = torch.from_numpy(image).float()
            label = torch.from_numpy(label).reshape(-1)

            # print(f'input shape = {image.shape}')
            pred = model(image)

            loss = loss_fn(pred,label)

            optim.zero_grad()
            loss.backward()
            optim.step()

            training_loss[e] += loss.item()
            pred = pred.detach().numpy()
            label = label.detach().numpy()
            # print(pred.shape)
            # print(label.shape)
            # exit(1)
            training_accuracy[e] += ave_accuracy_torch(pred, label)
            # print(training_accuracy[e].shape)

        # At each epoch 'e', print training loss and accuracy (homework)
        training_loss[e] = training_loss[e] / num_train_batches
        training_accuracy[e] = training_accuracy[e] / num_train_batches
        print(f'{e}th training loss = {training_loss[e]}\n{e}th training accuracy = {training_accuracy[e]}')
