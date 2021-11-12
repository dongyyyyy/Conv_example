import numpy as np

class Conv2d():
    def __init__(self,in_channels,out_channels,kernel_size=(3,3),stride=(1,1),padding=(0,0),dilation=(0,0),padding_mode='zero'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding_mode = padding_mode

        # check tuple
        self.kernel_size = self.check_tuple(kernel_size)
        self.stride = self.check_tuple(stride)
        self.padding = self.check_tuple(padding)
        self.dilation = self.check_tuple(dilation)

        init_coef = 1
        # output channel, input channel, kernelsize , kernsize
        self.conv_w = init_coef * np.random.randn(in_channels * self.kernel_size[0] * self.kernel_size[1], out_channels)

        self.conv_b = init_coef * np.random.randn(out_channels)

    def check_tuple(self, x):
        if type(x) == type(0):
            x = (x, x)
        elif len(x) != 2:
            assert len(x) != 2, 'tuple size is over 2!!!'
        return np.array(x)

    def im2col(self,inputs):
        self.output_height = (inputs.shape[-2] - self.kernel_size[0] + (2* self.padding[0])) // self.stride[0] + 1
        self.output_width = (inputs.shape[-1] - self.kernel_size[1] + (2 * self.padding[1])) // self.stride[1] + 1
        # input size = [1, 3, 5,5]
        # kernel size = [5, 3, 3,3]
        # (3*3*3, 5) = (27,5)
        im2col_in = np.zeros((inputs.shape[0],self.output_height *
                              self.output_width,self.in_channels *
                              self.kernel_size[0]*self.kernel_size[1]))

        # 1,3,5,5 => [1,25,27]
        if self.padding[0] > 0:
            inputs_padding = np.zeros((inputs.shape[0],inputs.shape[1],inputs.shape[-2]+(2*self.padding[0]),inputs.shape[-1]+(2*self.padding[1])))
            inputs_padding[:,:,self.padding[0]:inputs.shape[-2]+self.padding[0],self.padding[1]:inputs.shape[-1]+self.padding[1]] = inputs
            inputs = inputs_padding
        else:
            inputs = inputs

        for batch in range(inputs.shape[0]):
            for channels in range(inputs.shape[1]):
                im2col_height = 0
                for height in range(0,self.output_height):
                    for width in range(0,self.output_width):
                        im2col_in[batch,im2col_height,
                        (channels*self.kernel_size[0]*self.kernel_size[1]):((channels+1)*self.kernel_size[0]*self.kernel_size[1])] = \
                            inputs[batch, channels, (height*+self.stride[0]*1):(height*+self.stride[0]*1)+self.kernel_size[0], (width*self.stride[1]*1):width*self.stride[1]+self.kernel_size[1]].reshape(1,-1)
                        im2col_height += 1

        return im2col_in

    def calcul_matmul(self,inputs):
        im2col_out = np.matmul(inputs, np.expand_dims(self.conv_w.copy(),axis=0)) + self.conv_b
        # print(f'np.expand_dims(self.conv_w,axis=0) = {self.conv_w.shape}')
        # np.expand_dims(self.conv_w, axis=0)

        im2col_out = np.transpose(im2col_out,(0,2,1))
        im2col_out = im2col_out.reshape(inputs.shape[0],self.out_channels,self.output_height,self.output_width)
        return im2col_out

    def forward(self,x):
        self.x = self.im2col(x) # z^[l](x_l-1)
        self.z = self.calcul_matmul(self.x)
        return self.z

    def backward(self):
        return self.conv_w

    def update(self,delta,alpha):
        dw = np.sum(np.matmul(np.transpose(self.x,(0,2,1)),delta),axis=0)/self.x.shape[0]
        db = np.matmul(np.ones((1,self.x.shape[0])),delta.sum(axis=1))/self.x.shape[0]

        self.conv_w = self.conv_w - alpha * dw
        self.conv_b = self.conv_b - alpha * db