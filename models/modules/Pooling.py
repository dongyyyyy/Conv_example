import numpy as np

def one_hot_max(y, length_of_onehot):
    return np.eye(length_of_onehot)[y]

class maxpool2d():
    def __init__(self,kernel_size=(2,2),stride=(2,2),padding=(0,0),padding_mode='zero'):
        super().__init__()
        self.kernel_size = self.check_tuple(kernel_size)
        self.stride = self.check_tuple(stride)
        self.padding = self.check_tuple(padding)

    def check_tuple(self, x):
        if type(x) == type(0):
            x = (x, x)
        elif len(x) != 2:
            assert len(x) != 2, 'tuple size is over 2!!!'
        return np.array(x)

    def forward(self,inputs):
        output_height = (inputs.shape[-2] - self.kernel_size[0] + (2 * self.padding[0])) // self.stride[0] + 1
        output_width = (inputs.shape[-1] - self.kernel_size[1] + (2 * self.padding[1])) // self.stride[1] + 1
        im2col_in = np.zeros((inputs.shape[0], inputs.shape[1], output_height * output_width, self.kernel_size[0]*self.kernel_size[1]))
        # im2col_out = np.zeros((inputs.shape[0], inputs.shape[1], output_height * output_width, 1))

        if self.padding[0] > 0:
            inputs_padding = np.zeros((inputs.shape[0],inputs.shape[1],inputs.shape[-2]+(2*self.padding[0]),inputs.shape[-1]+(2*self.padding[1])))
            inputs_padding[:,:,self.padding[0]:inputs.shape[-2]+self.padding[0],self.padding[1]:inputs.shape[-1]+self.padding[1]] = inputs
            inputs = inputs_padding
        else:
            inputs = inputs
        for batch in range(inputs.shape[0]):
            for channels in range(inputs.shape[1]):
                im2col_height = 0
                for height in range(0,output_height):
                    for width in range(0,output_width):
                        im2col_in[batch,channels,
                        im2col_height,:] = \
                            inputs[batch, channels, (height*self.stride[0]*1):(height*self.stride[0]*1)+self.kernel_size[0], (width*self.stride[1]*1):(width*self.stride[1]*1)+self.kernel_size[1]].reshape(1,-1)
                        im2col_height += 1

        im2col_out = np.max(im2col_in,axis=3)
        argmax = np.array(one_hot_max(np.argmax(im2col_in,axis=3),self.kernel_size[0]*self.kernel_size[1]),dtype=np.int32)
        # self.argmax = argmax
        # print(argmax)
        # return self.argmax

        self.argmax = np.zeros((inputs.shape[0],inputs.shape[1],inputs.shape[2],inputs.shape[3]))

        for batch in range(inputs.shape[0]):
            for channels in range(inputs.shape[1]):
                im2col_height = 0
                for height in range(0,output_height):
                    for width in range(0,output_width):
                        self.argmax[batch,channels,height*self.stride[0]:height*self.stride[0]+self.kernel_size[0],width*self.stride[1]:width*self.stride[1]+self.kernel_size[1]] = \
                            argmax[batch,channels,im2col_height,:].reshape(self.kernel_size[0],self.kernel_size[1])

                        im2col_height += 1

        out = im2col_out.reshape(inputs.shape[0],inputs.shape[1],output_height,output_width)
        return out

    def backward(self):
        return self.argmax



class GlobalAveragePooling2d():
    def __init__(self):
        super().__init__()


    def forward(self,inputs):
        kernel_size = inputs.shape[-1]

        out = np.mean(np.mean(inputs,axis=-1),axis=-1)

        self.back = np.ones(inputs.shape)

        self.back = self.back / (kernel_size*kernel_size)

        return out

    def backward(self):
        return self.back

