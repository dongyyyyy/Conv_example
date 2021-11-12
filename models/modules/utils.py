import numpy as np

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=1).reshape(-1, 1)

def one_hot(y, length_of_onehot):
    return np.eye(length_of_onehot)[y].reshape(-1, length_of_onehot)


class crossentropy():
    def __init__(self,class_num=10):
        super().__init__()
        self.class_num = class_num

    def forward(self,z, y):
        y = (one_hot(y,self.class_num))
        self.z = softmax(z)
        self.y = y
        # print(self.z.shape)
        # print(self.y.shape)
        return np.mean(-1 * np.sum(self.y * np.log(self.z), axis=1))

    def backward(self):
        return self.z - self.y


class ReLU():
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.x = x
        return np.maximum(0,x)

    def backward(self):
        return np.where(self.x >0, 1, 0)

    def backward_conv(self,kernel_size):
        return np.transpose(np.where(self.x>0,1,0).reshape(self.x.shape[0],self.x.shape[1],-1),axes=(0,2,1))