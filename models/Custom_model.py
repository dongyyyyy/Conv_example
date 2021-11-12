from models.modules.Conv import *
from models.modules.Linear import *
from models.modules.Pooling import *
from models.modules.utils import *

class Custom_models():
    def __init__(self,in_channels=3,layer_channels=[16,32,32,64,128],class_num=10,alpha=0.001):
        self.alpha = alpha
        self.conv1 = Conv2d(in_channels=in_channels,out_channels=layer_channels[0],kernel_size=3,stride=1,padding=1)
        self.relu1 = ReLU()
        self.maxpool1 = maxpool2d(kernel_size=2,stride=2,padding=0)

        self.conv2 = Conv2d(in_channels=layer_channels[0], out_channels=layer_channels[1],kernel_size=3,stride=1,padding=1)
        self.relu2 = ReLU()
        # self.maxpool2 = maxpool2d(kernel_size=2, stride=2, padding=0)

        self.avg = GlobalAveragePooling2d()
        self.fc = linears(in_channels=layer_channels[1], out_channels=class_num)

    def forward(self,x):
        # print(x.shape)
        out = self.conv1.forward(x)
        # print(out.shape)
        out = self.relu1.forward(out)
        out = self.maxpool1.forward(out)
        # print(out.shape)

        out = self.conv2.forward(out)
        # print(out.shape)
        out = self.relu2.forward(out)
        # out = self.maxpool1.forward(out)

        out = self.avg.forward(out)
        # print(out.shape)
        out = self.fc.forward(out)
        # print(out.shape)

        return out

    def backward(self,o):
        self.delta3 = o # for FC

        self.delta2 = np.expand_dims(np.expand_dims(np.matmul(self.delta3,self.fc.backward().copy().T),axis=-1),axis=-1)*self.avg.backward()*self.relu2.backward() # for Conv2
        self.delta2 = self.delta2.reshape(self.delta2.shape[0],self.delta2.shape[1],-1)
        self.delta2 = np.transpose(self.delta2,(0,2,1))

        # self.delta2 = self.delta2.reshape(self.delta2.shape[0], self.delta2.shape[1], -1)
        conv_w2 = np.transpose(self.conv2.backward().copy(),axes=(1,0)).reshape(self.conv2.out_channels,self.conv2.in_channels,self.conv2.kernel_size[0],self.conv2.kernel_size[1])
        conv_w2 = np.sum(conv_w2,axis=-2).sum(axis=-1)

        delta1_w = np.transpose(np.matmul(self.delta2, conv_w2), axes=(0, 2, 1))

        # [b,25,32] matmul [32,144] => [b,25,144]
        delta1_w = np.repeat(delta1_w.reshape(delta1_w.shape[0],delta1_w.shape[1],int(np.sqrt(delta1_w.shape[2])),int(np.sqrt(delta1_w.shape[2]))),self.maxpool1.kernel_size[0],axis=-2)
        delta1_w = np.repeat(delta1_w,self.maxpool1.kernel_size[1], axis=-1)


        self.delta1 = delta1_w*self.maxpool1.backward()*self.relu1.backward() # for Conv1
        self.delta1 = self.delta1.reshape(self.delta1.shape[0], self.delta1.shape[1], -1)
        self.delta1 = np.transpose(self.delta1, (0, 2, 1))

    def update(self):
        self.fc.update(self.delta3,self.alpha)

        self.conv2.update(self.delta2.copy(),self.alpha)

        self.conv1.update(self.delta1,self.alpha)