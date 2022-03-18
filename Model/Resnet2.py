import torch
import torch.nn as nn
from HS_block import HSBlock


def conv1x1(in_channel, out_channel):
    return nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=(1, 1), bias=False),
                         nn.BatchNorm2d(out_channel)
                         )


def conv3x1(in_channel, out_channel):
    return nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=(8, 1), stride=(2, 1), padding=(0, 0), bias=False),
                         nn.BatchNorm2d(out_channel))


class Resnet2(nn.Module):

    def __init__(self, out_channel, split, basic_channel):
        super(Resnet2, self).__init__()

        # self.first_conv = conv3x3(in_channel, out_channel)

        self.HS_conv = HSBlock(out_channel, split, basic_channel)

        if out_channel % split == 0:
            self.last_conv = conv1x1((out_channel//split + basic_channel + (split-2)*(basic_channel//2)), out_channel)
        else:
            self.last_conv = conv1x1((out_channel//split + 1  + basic_channel + (split-2)*(basic_channel//2)), out_channel)

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
        # out = self.relu(out)
        return out

#
# if __name__ == '__main__':
#     x = torch.randn(1, 64, 171, 40)
#     model = Resnet2(64, 4, 18)
#     # print(model)
#     print(model(x).shape)








