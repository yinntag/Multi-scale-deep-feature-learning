import torch
import torch.nn as nn
from GF import HSBlock


def conv1x1(in_channel, out_channel):
    return nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=(1, 1), bias=False),
                         nn.BatchNorm2d(out_channel)
                         )


def conv3x1(in_channel, out_channel):
    return nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=(8, 1), stride=(2, 1), padding=(0, 0), bias=False),
                         nn.BatchNorm2d(out_channel))


class GF_CNN(nn.Module):

    def __init__(self, out_channel, split, basic_channel, mode='GF_4'):
        super(GF_CNN, self).__init__()
        self.mode = mode if mode in ['GF_1', 'GF_2', 'GF_3', 'GF_4'] else 'GF_4'
        # self.first_conv = conv3x3(in_channel, out_channel)
        self.HS_conv = HSBlock(out_channel, split, basic_channel, mode='GF_4')

        if self.mode == 'GF_1':
            self.last_conv = conv1x1(out_channel//split, out_channel)
        elif self.mode == 'GF_2':
            self.last_conv = conv1x1(out_channel//split + basic_channel//2, out_channel)
        elif self.mode == 'GF_3':
            self.last_conv = conv1x1(out_channel//split + (split-2)*(basic_channel//2), out_channel)
        elif self.mode == 'GF_4':
            self.last_conv = conv1x1((out_channel//split + basic_channel + (split-2)*(basic_channel//2)), out_channel)

        # self.shortcut_conv = conv3x3(in_channel, out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # out = self.first_conv(x)
        # print(out.shape, 'out1')
        out = self.HS_conv(x)
        # print(out.shape, 'out2')
        out = self.last_conv(out)
        # y = self.shortcut_conv(x)
        # out += y
        out = self.relu(out)
        return out


class GF_ResNet(nn.Module):

    def __init__(self, out_channel, split, basic_channel, mode='GF_4'):
        super(GF_ResNet, self).__init__()
        self.mode = mode if mode in ['GF_1', 'GF_2', 'GF_3', 'GF_4'] else 'GF_4'
        # self.first_conv = conv3x3(in_channel, out_channel)
        self.HS_conv = HSBlock(out_channel, split, basic_channel, mode='GF_4')

        if self.mode == 'GF_1':
            self.last_conv = conv1x1(out_channel//split, out_channel)
        elif self.mode == 'GF_2':
            self.last_conv = conv1x1(out_channel//split + basic_channel//2, out_channel)
        elif self.mode == 'GF_3':
            self.last_conv = conv1x1(out_channel//split + (split-2)*(basic_channel//2), out_channel)
        elif self.mode == 'GF_4':
            self.last_conv = conv1x1((out_channel//split + basic_channel + (split-2)*(basic_channel//2)), out_channel)

        # self.shortcut_conv = conv3x3(in_channel, out_channel)

    def forward(self, x):
        # out = self.first_conv(x)
        # print(out.shape, 'out1')
        out = self.HS_conv(x)
        # print(out.shape, 'out2')
        out = self.last_conv(out)
        # y = self.shortcut_conv(x)
        # out += y
        return out

#
# if __name__ == '__main__':
#     x = torch.randn(1, 64, 171, 40)
#     model = Resnet2(64, 4, 28)
#     # print(model)
#     print(model(x).shape)








