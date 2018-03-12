# !/usr/bin/env python
# coding=utf-8
from torch.autograd import Variable
from collections import OrderedDict
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import math

__all__ = ['RankingDenseNet',
           'ranking_resnet_18', 'ranking_resnet_34', 'ranking_resnet_50', 'ranking_resnet_101', 'ranking_resnet_151']


def initialize_weights(layer_modules):
    for m in layer_modules:
        if isinstance(m, nn.Conv2d):
            num = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / num))
            # m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv2d):
            num = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / num))
            # m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


def conv3x3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class RankingResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        self.inplanes = 64
        super(RankingResNet, self).__init__()

        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)),
            ('bn1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ('layer1', self._make_layer(block, 64, layers[0])),
            ('layer2', self._make_layer(block, 128, layers[1], stride=2)),
            ('layer3', self._make_layer(block, 256, layers[2], stride=2)),
            ('layer4', self._make_layer(block, 512, layers[3], stride=2)),
        ]))

        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.attribute_feature_fc = nn.Linear(512 * block.expansion, 256)

        self.attribute_subtlety_score_fc = nn.Linear(256, 1)
        self.attribute_internalStructure_score_fc = nn.Linear(256, 1)
        self.attribute_calcification_score_fc = nn.Linear(256, 1)
        self.attribute_sphericity_score_fc = nn.Linear(256, 1)
        self.attribute_margin_score_fc = nn.Linear(256, 1)
        self.attribute_lobulation_score_fc = nn.Linear(256, 1)
        self.attribute_spiculation_score_fc = nn.Linear(256, 1)
        self.attribute_texture_score_fc = nn.Linear(256, 1)

        self.dropout = nn.Dropout2d(0.5)

        self.classifier = nn.Linear(512 * block.expansion + 256, num_classes)

        # self.features = nn.Sequential(OrderedDict([
        #           ('conv1', self.conv1),
        #           ('bn1', self.bn1),
        #           ('conv1relu', self.relu),
        #           ('maxpool', self.maxpool),
        #           ('layer1', self.layer1),
        #           ('layer2', self.layer2),
        #           ('layer3', self.layer3),
        #           ('layer4', self.layer4),
        #           ('avgpool', self.avgpool),
        #         ])
        # )
        #
        # self.classifier_malignacy = nn.Sequential(OrderedDict([
        #           ('score_fc', self.score_fc)])
        # )
        #
        # self.classifier_attribute = nn.Sequential(OrderedDict([
        #           ('attribute_feature_fc', self.attribute_feature_fc),
        #
        #           ('attribute_subtlety_score_fc', self.attribute_subtlety_score_fc),
        #           ('attribute_internalStructure_score_fc', self.attribute_internalStructure_score_fc),
        #           ('attribute_calcification_score_fc', self.attribute_calcification_score_fc),
        #           ('attribute_sphericity_score_fc', self.attribute_sphericity_score_fc),
        #           ('attribute_margin_score_fc', self.attribute_margin_score_fc),
        #           ('attribute_lobulation_score_fc', self.attribute_lobulation_score_fc),
        #           ('attribute_spiculation_score_fc', self.attribute_spiculation_score_fc),
        #           ('attribute_texture_score_fc', self.attribute_texture_score_fc),
        #         ])
        # )

        initialize_weights(self.modules())

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward_once(self, x):
        out = self.features(x)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)

        attribute_feature = self.attribute_feature_fc(out)
        subtlety_score = self.attribute_subtlety_score_fc(attribute_feature)
        internalStructure_score = self.attribute_internalStructure_score_fc(attribute_feature)
        calcification_score = self.attribute_calcification_score_fc(attribute_feature)
        sphericity_score = self.attribute_sphericity_score_fc(attribute_feature)
        margin_score = self.attribute_margin_score_fc(attribute_feature)
        lobulation_score = self.attribute_lobulation_score_fc(attribute_feature)
        spiculation_score = self.attribute_spiculation_score_fc(attribute_feature)
        texture_score = self.attribute_texture_score_fc(attribute_feature)

        out = torch.cat([attribute_feature, out], dim=1)

        out = self.dropout(out)

        out = self.classifier(out)

        return out, [subtlety_score, internalStructure_score, calcification_score, sphericity_score, margin_score, lobulation_score, spiculation_score, texture_score]
        # return out, [subtlety_score, internalStructure_score, calcification_score, sphericity_score,
        #              margin_score, lobulation_score, spiculation_score, texture_score]

    def forward(self, input):
        # output_1, attribute_score_1 = self.forward_once(input)
        #
        # return output_1, attribute_score_1

        input_1 = input[0:int(input.shape[0]/2), :, :, :]
        print input_1.shape
        input_2 = input[int(input.shape[0]/2):input.shape[0], :, :, :]
        output_1, attribute_score_1 = self.forward_once(input_1)
        output_2, attribute_score_2 = self.forward_once(input_2)

        cat_output = torch.cat([output_1, output_2])
        cat_subtlety_score = torch.cat([attribute_score_1[0], attribute_score_2[0]])
        cat_internalStructure_score = torch.cat([attribute_score_1[1], attribute_score_2[1]])
        cat_calcification_score = torch.cat([attribute_score_1[2], attribute_score_2[2]])
        cat_sphericity_score = torch.cat([attribute_score_1[3], attribute_score_2[3]])
        cat_margin_score = torch.cat([attribute_score_1[4], attribute_score_2[4]])
        cat_lobulation_score = torch.cat([attribute_score_1[5], attribute_score_2[5]])
        cat_spiculation_score = torch.cat([attribute_score_1[6], attribute_score_2[6]])
        cat_texture_score = torch.cat([attribute_score_1[7], attribute_score_2[7]])

        return cat_output, \
               cat_subtlety_score, cat_internalStructure_score, \
               cat_calcification_score, cat_sphericity_score, \
               cat_margin_score, cat_lobulation_score, \
               cat_spiculation_score, cat_texture_score


def ranking_resnet_18(**kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = RankingResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

    return model


def ranking_resnet_34(**kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = RankingResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

    return model


def ranking_resnet_50(**kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = RankingResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    return model


def ranking_resnet_101(**kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = RankingResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

    return model


def ranking_resnet_152(**kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = RankingResNet(Bottleneck, [3, 8, 36, 3], **kwargs)

    return model


def get_loss(output_score_1,
             cat_subtlety_score, cat_internalStructure_score, cat_calcification_score, cat_sphericity_score,
             cat_margin_score, cat_lobulation_score, cat_spiculation_score, cat_texture_score,
             gt_score_1,
             gt_attribute_score_1):
    print output_score_1.size(), cat_subtlety_score.size(), \
        gt_score_1.size(), gt_attribute_score_1.size()
    print type(output_score_1.data), type(cat_subtlety_score.data), \
        type(gt_score_1.data), type(gt_attribute_score_1.data)
    # basic loss
    xcentloss_func_1 = nn.CrossEntropyLoss()
    xcentloss_1 = xcentloss_func_1(output_score_1, gt_score_1)

    # ranking loss
    ranking_loss_sum = 0
    half_size_of_output_score = output_score_1.size()[0] / 2
    for i in range(half_size_of_output_score):
        tmp_output_1 = output_score_1[i]
        tmp_output_2 = output_score_1[i + half_size_of_output_score]
        tmp_gt_score_1 = gt_score_1[i]
        tmp_gt_score_2 = gt_score_1[i + half_size_of_output_score]

        rankingloss_func = nn.MarginRankingLoss()

        if tmp_gt_score_1.data[0] > tmp_gt_score_2.data[0]:
            # print ">", gt_score_1, gt_score_2
            target = torch.ones(1) * -1
            ranking_loss_sum += rankingloss_func(tmp_output_1, tmp_output_2, Variable(target.cuda()))
        else:
            # print "<", gt_score_1, gt_score_2
            target = torch.ones(1)
            ranking_loss_sum += rankingloss_func(tmp_output_1, tmp_output_2, Variable(target.cuda()))

    ranking_loss = ranking_loss_sum / half_size_of_output_score

    # attribute loss
    attribute_mseloss_func_1 = nn.CrossEntropyLoss()
    attribute_mseloss_1 = attribute_mseloss_func_1(cat_subtlety_score, gt_attribute_score_1)

    loss = 0.6 * xcentloss_1 + 0.2 * ranking_loss + 0.2 * attribute_mseloss_1

    return loss


def main():
    # siamese number, batch size, channel, x, y
    siamese_number = 2
    batch_size = 2
    a = np.random.randn(siamese_number * batch_size, 3, 224, 224)
    tensor_5d = Variable(torch.from_numpy(a).float().cuda())
    # print "a:", tensor_5d.size()

    model = ranking_resnet_50(num_classes=2).cuda()
    # print model

    output_1, \
    cat_subtlety_score, cat_internalStructure_score, cat_calcification_score, cat_sphericity_score, \
    cat_margin_score, cat_lobulation_score, cat_spiculation_score, cat_texture_score = model(tensor_5d)

    gt_output_1 = np.zeros(siamese_number * batch_size)
    gt_output_1 = Variable(torch.from_numpy(gt_output_1).long().cuda())

    gt_attribute_score_1 = np.zeros(siamese_number * batch_size)
    gt_attribute_score_1 = Variable(torch.from_numpy(gt_attribute_score_1).long().cuda())

    loss = get_loss(output_1,
                    cat_subtlety_score, cat_internalStructure_score, cat_calcification_score, cat_sphericity_score,
                    cat_margin_score, cat_lobulation_score, cat_spiculation_score, cat_texture_score,
                    gt_output_1,
                    gt_attribute_score_1)

    print loss


if __name__ == '__main__':
    main()
