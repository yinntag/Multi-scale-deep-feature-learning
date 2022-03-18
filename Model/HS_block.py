import torch
import torch.nn as nn


class HSBlock(nn.Module):
    def __init__(self, in_planes, s, w):
        super(HSBlock, self).__init__()
        self.s = s
        self.module_list = nn.ModuleList()
        if in_planes % s == 0:
            in_ch, in_ch_last = in_planes // s, in_planes // s
        else:
            in_ch, in_ch_last = (in_planes // s) + 1, in_planes - (in_planes // s + 1) * (s-1)
            # print(in_ch, in_ch_last)
        for i in range(self.s):
            if i == 0:
                self.module_list.append(nn.Sequential())
            elif i == 1:
                self.module_list.append(self.conv_bn_relu(in_ch=in_ch, out_ch=w))
            elif i == s - 1:
                self.module_list.append(self.conv_bn_relu(in_ch=in_ch_last + w // 2, out_ch=w))
            else:
                self.module_list.append(self.conv_bn_relu(in_ch=in_ch + w // 2, out_ch=w))
        self.initialize_weights()

    def conv_bn_relu(self, in_ch, out_ch, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)):
        conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        return conv_bn_relu

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        # print(x.shape, 'x')
        x = list(x.chunk(chunks=self.s, dim=1))
        # ttt = x[1]
        # print(ttt)
        # print(x, 'xxxx')
        for i in range(1, len(self.module_list)):
            # print(i, 'iii')
            # print(self.module_list[i], '11111')
            y = self.module_list[i](x[i])
            # print(y.shape)
            if i == len(self.module_list) - 1:
                # print(i, 'iiiiiiii')
                x[0] = torch.cat((x[0], y), 1)
            else:
                y1, y2 = y.chunk(chunks=2, dim=1)
                x[0] = torch.cat((x[0], y1), 1)
                x[i + 1] = torch.cat((x[i + 1], y2), 1)
            # print(x[0].shape)
        return x[0]


# if __name__ == '__main__':
#     x = torch.randn(1, 64, 171, 40)
#     model = HSBlock(64, 4, 28)
#     print(model)
#     print(model(x).shape)
