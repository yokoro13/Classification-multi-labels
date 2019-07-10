import torch.nn as nn
import numpy as np


class Net(nn.Module):
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Net, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        layers.append(nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        out_cls = self.main(x)
        return out_cls.view(out_cls.size(0), out_cls.size(1))