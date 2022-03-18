import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as Data
from torch.autograd import Variable
import os
from hs_cnn import hs_cnn
from Resnet2 import Resnet2
import matplotlib.pyplot as plt
import pylab as pl
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()

train_x = np.load('./pamap2/train_x.npy')
train_x = torch.from_numpy(np.reshape(train_x.astype(float),
                                      [train_x.shape[0], 1,
                                       train_x.shape[1],
                                       train_x.shape[2]])).type(torch.FloatTensor).cuda()

train_y = np.load('./pamap2/train_y.npy')
train_y = torch.from_numpy(train_y).type(torch.FloatTensor).cuda()

test_x = np.load('./pamap2/test_x.npy')
test_x = torch.from_numpy(np.reshape(test_x.astype(float),
                                     [test_x.shape[0], 1,
                                      test_x.shape[1],
                                      test_x.shape[2]])).type(torch.FloatTensor).cuda()


test_y = np.load('./pamap2/test_y.npy')
test_y = torch.from_numpy(test_y.astype(np.float32)).type(torch.FloatTensor).cuda()

torch_dataset = Data.TensorDataset(train_x, train_y)
train_loader = Data.DataLoader(dataset=torch_dataset, batch_size=200, shuffle=True, num_workers=0)
torch_dataset = Data.TensorDataset(test_x, test_y)
test_loader = Data.DataLoader(dataset=torch_dataset, batch_size=200, shuffle=True, num_workers=0)


class HS_CNN(nn.Module):
    def __init__(self):
        super(HS_CNN, self).__init__()
        self.Block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            hs_cnn(64, 4, 28)
        )
        self.Block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            hs_cnn(128, 4, 28)
        )
        self.Block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            hs_cnn(256, 4, 28)
        )
        self.fc = nn.Sequential(
            nn.Linear(51200, 12)
        )

    def forward(self, x):
        # print(x.shape)
        out = self.Block1(x)
        out = self.Block2(out)
        out = self.Block3(out)
        out = out.view(out.size(0), -1)
        # print(out.shape)
        out = self.fc(out)
        # print(out.shape)
        out = nn.LayerNorm(out.size())(out.cpu())
        # print(out.shape)
        out = out.cuda()
        return out

class HS_ResNet(nn.Module):
    def __init__(self):
        super(HS_ResNet, self).__init__()
        self.Block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            Resnet2(64, 6, 28)
        )
        self.shortcut1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
        )

        self.Block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            Resnet2(128, 6, 28)
        )
        self.shortcut2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
        )

        self.Block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            Resnet2(256, 6, 28)
        )
        self.shortcut3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(256),
        )
        self.fc = nn.Sequential(
            nn.Linear(51200, 12)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # print(x.shape)
        out1 = self.Block1(x)
        y1 = self.shortcut1(x)
        out = y1 + out1
        out = self.relu(out)

        out2 = self.Block2(out)
        y2 = self.shortcut2(out)
        out = y2 + out2
        out = self.relu(out)

        out3 = self.Block3(out)
        y3 = self.shortcut3(out)
        out = y3 + out3
        out = self.relu(out)

        out = out.view(out.size(0), -1)
        # print(out.shape)
        out = self.fc(out)
        print(out.shape)
        out = nn.LayerNorm(out.size())(out.cpu())
        print(out.shape)
        out = out.cuda()
        return out

model = CNN().cuda()
print(model)

# learning_rate = 0.001
learning_rate = 5e-4


def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train(epoch):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total_Number of params: {} |Trainable_num of params: {}'.format(total_num, trainable_num))
    adjust_learning_rate(optimizer, epoch)
    global cur_batch_win
    model.train()
    loss_list, batch_list = [], []
    for i, (images, labels) in enumerate(train_loader):
        images, labels = Variable(images).cuda(), Variable(labels).cuda()
        labels = labels.long().cpu()
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output.cpu(), labels.cpu())
        loss_list.append(loss.data.item())
        batch_list.append(i + 1)

        if i == 1:
            print('Current Learning_rate: ', optimizer.param_groups[0]['lr'])
            print('Training: Epoch %d,  Loss: %f' % (epoch, loss.data.item()))

        loss.backward()
        optimizer.step()


acc = 0
acc_best = 0


def test(test_acc):
    global acc, acc_best
    model.eval()
    total_correct = 0
    avg_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
            labels = labels.long().cpu()
            start = time.perf_counter()
            output = model(images)
            end = time.perf_counter()
            avg_loss += criterion(output.cpu(), labels).cuda().sum()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.cuda().data.view_as(pred)).cuda().sum()

    accuracy = accuracy_score(labels.cpu().numpy(), pred.cpu().numpy())
    f1 = f1_score(labels.cpu().numpy(), pred.cpu().numpy(), average='weighted')
    precision = precision_score(labels.cpu().numpy(), pred.cpu().numpy(), average='weighted')
    recall = recall_score(labels.cpu().numpy(), pred.cpu().numpy(), average='weighted')

    print('test accuracy: %.4f' % accuracy,
          '| test F1: %.4f' % f1,
          '| test precision: %.4f' % precision,
          '| test recall: %.4f' % recall)

    avg_loss /= len(test_x)
    acc = float(total_correct) / len(test_x)
    if acc_best < acc:
        acc_best = acc

def train_and_test(epoch):
    train(epoch)
    test(test_acc)


test_acc = []


def main():
    epoch = 200
    for e in range(0, epoch):
        train_and_test(e)


if __name__ == '__main__':
    main()


