import torch as t
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import math


# part 1.
class BasicBlock_TestGroup(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels,
                 size_x=3, size_y=3, size_z=3,
                 stride_x=1, stride_y=1, stride_z=1,
                 downsample=None):
        super(BasicBlock_TestGroup, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(size_x, size_y),
                               stride=(stride_x, stride_z), padding=(1, 1), bias=False, groups=3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data = t.ones(m.weight.data.shape)
                # print m.weight.data

    def forward(self, x):
        out = self.conv1(x)

        return out


class BasicBlock_2D_Prototype(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, shortcut_downsample=None):
        super(BasicBlock_2D_Prototype, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False, groups=in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # self.shortcut_downsample = shortcut_downsample
        self.shortcut_downsample = self.get_shortcut_downsample(in_channels, out_channels, stride=stride)
        self.stride = stride

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                m.weight.data = t.ones(m.weight.data.shape)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def get_shortcut_downsample(self, in_channels, out_channels, stride=1):
        if stride != 1 or in_channels != out_channels:
            shortcut_downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            return shortcut_downsample
        else:
            return None

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        shortcut = x
        if self.shortcut_downsample is not None:
            shortcut = self.shortcut_downsample(x)

        out += shortcut
        out = self.relu(out)

        return out

        # print "x~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        # print x.data
        # out = self.conv1(x)
        # print "conv1~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        # print out.data
        # out = self.bn1(out)
        # print "bn1~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        # print out.data
        # out = self.relu(out)
        # print "relu~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        # print out.data
        #
        # out = self.conv2(out)
        # print "conv2~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        # print out.data
        # out = self.bn2(out)
        # print "bn2~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        # print out.data
        # out = self.relu(out)
        # print "relu~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        # print out.data
        #
        # return out


class BasicBlock_3D_Prototype(nn.Module):
    def __init__(self, sub_block, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 shortcut_downsample=None):
        super(BasicBlock_3D_Prototype, self).__init__()

        basic_2d_bloscks = []
        for i in range(in_channels):
            basic_2d_bloscks.append(self._make_basic_2d_block(sub_block, in_channels, in_channels,
                                                              kernel_size=3, stride=1, padding=1))

        self.basic_2d_bloscks = nn.ModuleList(basic_2d_bloscks)

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False, groups=in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # self.shortcut_downsample = shortcut_downsample
        self.shortcut_downsample = self.get_shortcut_downsample(in_channels, out_channels, stride=stride)
        self.stride = stride

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                m.weight.data = t.ones(m.weight.data.shape)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_basic_2d_block(self, sub_block, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        return sub_block(in_channels, out_channels, kernel_size, stride, padding)

    def get_shortcut_downsample(self, in_channels, out_channels, stride=1):
        if stride != 1 or in_channels != out_channels:
            shortcut_downsample = nn.SequentHiial(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            return shortcut_downsample
        else:
            return None

    def forward(self, x):
        # print "x~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        # print x.data
        out = self.conv1(x)
        # print "conv1~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        # print out.data
        out = self.bn1(out)
        # print "bn1~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        # print out.data
        out = self.relu(out)
        # print "relu~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        # print out.data

        out = self.conv2(out)
        # print "conv2~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        # print out.data
        out = self.bn2(out)
        # print "bn2~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        # print out.data

        shortcut = x
        if self.shortcut_downsample is not None:
            shortcut = self.shortcut_downsample(x)
        # print "shortcut~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        # print shortcut

        out += shortcut
        out = self.relu(out)

        return out


class BasicBlock_Linear(nn.Module):
    def __init__(self, sub_2d_block):
        super(BasicBlock_Linear, self).__init__()

        # basic_2d_bloscks = []
        # for i in range(10):
        #     basic_2d_bloscks.append(self._make_basic_2d_block(sub_2d_block, 5, kernel_size=3, stride=1, padding=1))
        # self.basic_2d_bloscks = nn.ModuleList(self.basic_2d_bloscks)

        self.basic_2d_bloscks = nn.ModuleList([nn.Linear(2, 2, bias=False) for i in range(10)])

        i = 0
        for m in self.modules():
            if isinstance(m, nn.Linear):
                i += 1
                m.weight.data = t.ones(m.weight.data.shape) * i

    def _make_basic_2d_block(self, sub_block, in_channels, kernel_size=3, stride=1, padding=1):
        return sub_block(in_channels, in_channels, kernel_size, stride, padding)

    def forward(self, x):
        z = t.stack([self.basic_2d_bloscks[i](x[:, i, :, :, :]) for i in range(x.size()[1])], 1)

        return z


# part 2. verify backward

def initialize_weight(layer_modules):
    for m in layer_modules:
        if isinstance(m, nn.Conv2d):
            num = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / num))
            # m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Conv3d):
            num = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * \
                  m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / num))
            # m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


# def initialize_weight_ones(layer_modules):
#     for m in layer_modules:
#         if isinstance(m, nn.Conv2d):
#             num = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             m.weight.data.fill_(1)
#             m.bias.data.zero_()
#         elif isinstance(m, nn.Conv3d):
#             num = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * \
#                   m.out_channels
#             m.weight.data.fill_(1)
#             m.bias.data.zero_()
#         elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
#             m.weight.data.fill_(1)
#             m.bias.data.zero_()


def test_initialization(layer_modules):
    for m in layer_modules:
        if isinstance(m, nn.Conv3d):
            print "w", m.weight.data
            print "b", m.bias.data


class Basic_2d_Block(nn.Module):
    """
    Note: without residual mechanism

    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Basic_2d_Block, self).__init__()

        self.seperate_conv2d = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, groups=in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2d_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.stride = stride

        initialize_weight(self.modules())

    def forward(self, x):
        out = self.seperate_conv2d(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2d_1x1(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class Basic_3d_Block(nn.Module):
    def __init__(self,
                 sub_2d_block, in_channels, out_channels,
                 sub_2d_in_channels, sub_2d_out_channels,
                 kernel_size=3, stride=1, padding=1):
        super(Basic_3d_Block, self).__init__()

        basic_2d_blocks = []
        for i in range(in_channels):
            basic_2d_blocks.append(self._make_basic_2d_block(sub_2d_block, sub_2d_in_channels, sub_2d_out_channels,
                                                             kernel_size=kernel_size, stride=stride, padding=padding))
        self.basic_2d_blocks = nn.ModuleList(basic_2d_blocks)

        self.conv3d_1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        initialize_weight(self.modules())

    @staticmethod
    def _make_basic_2d_block(sub_block, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        return sub_block(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        in_channels = x.size()[1]

        out = t.stack([self.basic_2d_blocks[i](x[:, i, :, :, :]) for i in range(in_channels)], 1)

        out = self.conv3d_1x1(out)
        out = self.bn(out)
        out = self.relu(out)


        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# test
def part_1():
    # part 1. verify conv layer
    a = np.resize(np.linspace(1, 96, 96).astype(np.float32), [2, 3, 4, 4])
    b1, b2 = a[0], a[1]
    b1 = np.expand_dims(b1, 0)
    b2 = np.expand_dims(b2, 0)

    a = Variable(t.from_numpy(a))
    b1 = Variable(t.from_numpy(b1))
    b2 = Variable(t.from_numpy(b2))

    print a, "\n\n\n\n", b1, "\n\n\n\n", b2

    model = BasicBlock_2D_Prototype(3, 6)
    score_a = model(a)
    print score_a.data.shape
    print score_a.data
    score_b1 = model(b1)
    score_b2 = model(b2)
    print score_b1.data.shape
    print score_b1.data
    print score_b2.data.shape
    print score_b2.data


def part_2():
    # part 2. verify backward
    a = np.resize(np.linspace(1, 1, 1250).astype(np.float32), [1, 1, 5, 5, 5])
    b = a + a
    c = b + a
    d = c + a
    e = np.concatenate([a, b], 1)
    # print "a:", a.shape, a
    # print "b:", b
    # print "c:", c
    # print "d:", d
    # print "e:", e.shape, e
    # print "e2:", e[:, 1, :, :].shape, e[:, 1, :, :]

    array_5d = e
    tensor_5d = Variable(t.from_numpy(array_5d))
    tensor_5d_shape = tensor_5d.size()
    in_channels = tensor_5d_shape[1]
    out_channels = 2
    sub_2d_in_channels = tensor_5d_shape[2]
    sub_2d_out_channels = tensor_5d_shape[2]
    print "in_channels:", in_channels
    print "out_channels:", out_channels
    print "sub_2d_in_channels:", sub_2d_in_channels
    print "sub_2d_out_channels:", sub_2d_out_channels

    model = Basic_3d_Block(Basic_2d_Block, in_channels, out_channels, sub_2d_in_channels, sub_2d_out_channels)
    score_a = model(tensor_5d)
    print score_a
    # print score_a
    # print "d:", type(tensor_5d), tensor_5d.size(), tensor_5d
    # print "score_a", type(score_a), score_a.size(), score_a

    # criterion = nn.CrossEntropyLoss()
    # x = criterion(score_a, Variable(t.from_numpy(np.resize(np.linspace(1, 20, 20), [1, 10, 1, 2]))))

    err = t.mean(t.abs(score_a - Variable(
        t.from_numpy(np.resize(np.linspace(1, 25000, 25000).astype(np.float32), [1, 2, 5, 5, 5])))))
    err.backward()

    optimizer = optim.Adam(model.parameters())

    optimizer.zero_grad()

    optimizer.step()


if __name__ == '__main__':
    part_2()
