import numpy as np


class linears():
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        init_coef = 1
        self.w = init_coef * np.random.randn(self.in_channels, self.out_channels)
        self.b = init_coef * np.random.randn(self.out_channels)

    def forward(self, x):
        self.x = x
        self.z = np.matmul(x, self.w) + self.b
        return self.z

    def backward(self):
        return self.w

    def update(self, delta, alpha):
        dw = np.matmul(self.x.T, delta) / self.x.shape[0]
        db = np.matmul(np.ones((1, self.x.shape[0])), delta) / self.x.shape[0]

        self.w = self.w - alpha * dw
        self.b = self.b - alpha * db
