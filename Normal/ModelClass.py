#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
from torch import nn
from torch.utils.data.dataset import Dataset
from PIL import Image
import numpy as np
from torchvision import models  # torchvision中提供了常用的深度学习模型
from PIL import ImageEnhance

'''  
# pytorch中数据通过Dataset进行封装，并通过DataLoader进行并行读取 
# Dataset: 对数据集的封装，提供索引方式的对数据样本进行读取
# DataLoder: 对Dataset进行封装，提供批量读取的迭代读取
'''
class SVHNDataset(Dataset):  # 对Dataset的继承
    # python类的构造函数,transform表示数据变换
    def __init__(self,img_path,img_label,transform=None):
        self.img_path = img_path
        self.img_label = img_label
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    # 获取样本对，模型直接对这一函数获得一对样本对
    def __getitem__(self,index):
        img = Image.open(self.img_path[index]).convert('RGB')
        # img = ImageEnhance.Contrast(img)
        # img.enhance(2.0)  # 所有图片对比度调到2

        if self.transform is not None:
            img = self.transform(img)

        # 原始SVHN中类别10为数字0
        lbl = []
        if len(self.img_label) > 0:
            lbl = [np.long(x) for x in self.img_label[index]]
        for i in range(4-len(lbl)):
            lbl.append(10)   # 补齐4个字符，用10填充，定长字符识别

        return img, torch.from_numpy(np.array(lbl)).long()  # 这部分有问题

    # 获取数据集长度
    def __len__(self):
        return len(self.img_path)

# 长度识别的Dataset
class LengthDataset(Dataset):  # 对Dataset的继承
    # python类的构造函数,transform表示数据变换
    def __init__(self,img_path,img_label,transform=None):
        self.img_path = img_path
        self.img_label = img_label
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    # 获取样本对，模型直接对这一函数获得一对样本对
    def __getitem__(self,index):
        img = Image.open(self.img_path[index]).convert('RGB')
        # img = ImageEnhance.Contrast(img)
        # img.enhance(2.0)  # 所有图片对比度调到2

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.from_numpy(np.array(self.img_label[index])).long()  # 这部分有问题

    # 获取数据集长度
    def __len__(self):
        return len(self.img_path)

'''    
# CNN是解决图像分类、图像检索、物体检测和语义分割的主流模型
# pytorch构建CNN非常简单，只需要定义好模型的参数和正向传播即可，pytorch会根据正向传播自动计算反向传播
# 定义一个简单的CNN模型
'''
class SVHN_EasyCNN_model(nn.Module):
    def __init__(self):
        super(SVHN_EasyCNN_model,self).__init__()  # 调用父类nn.Module的init()构造函数
        # 定义网络结构参数
        self.cnn = nn.Sequential(
            # in_channels=3(RGB image)、out_channels=16表示(3*16)个(3*3)的卷积核
            nn.Conv2d(3,16,kernel_size=(3,3),stride=(2,2)),
            # nn.BatchNorm2d(16),  # 参数为通道数
            nn.ReLU(),
            nn.Dropout(0.25),
            # 31*63
            nn.MaxPool2d(2), # kernel_size = 2
            # 15*31
            nn.Conv2d(16,32,kernel_size=(3,3),stride=(2,2)),
            # nn.BatchNorm2d(32),  # 参数为通道数
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool2d(2),
        )

        # 32个通道,每个通道3*7
        # 输出层11个,表示0~9以及没有数的概率值,并联4个fc层表示对每一位的预测
        self.fc1 = nn.Linear(32 * 3 * 7, 11)
        self.fc2 = nn.Linear(32 * 3 * 7, 11)
        self.fc3 = nn.Linear(32 * 3 * 7, 11)
        self.fc4 = nn.Linear(32 * 3 * 7, 11)

    def forward(self, img):  # 应该是重构了这个函数
        feat = self.cnn(img)
        # reshape到一维
        feat = feat.view(feat.shape[0], -1)
        # 并联4个全连接层
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)

        return [c1,c2,c3,c4]  # 表示每张图片预测的6个标签（6表示定长的长度）

# 使用resnet18的预训练模型
class SVHN_Resnet_model(nn.Module):
    def __init__(self):
        super(SVHN_Resnet_model, self).__init__()
        model_conv = models.resnet18(pretrained=True)  # 迁移学习
        model_conv.avgpool = nn.AdaptiveAvgPool2d(1)   # 修改resnet中的平均池化层参数
        model_conv = nn.Sequential(*list(model_conv.children())[:-1])
        self.cnn = model_conv

        self.fc1 = nn.Linear(512, 11)
        self.fc2 = nn.Linear(512, 11)
        self.fc3 = nn.Linear(512, 11)
        self.fc4 = nn.Linear(512, 11)

    def forward(self, img):  # 应该是重构了这个函数
        feat = self.cnn(img)
        # reshape到一维
        feat = feat.view(feat.shape[0], -1)
        # 并联4个全连接层
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)

        return [c1,c2,c3,c4]  # 表示每张图片预测的6个标签（6表示定长的长度）

# 预测图片中数字的长度
class Length_Recognition_model(nn.Module):
    def __init__(self):
        super(Length_Recognition_model,self).__init__()  # 调用父类nn.Module的init()构造函数

        model_conv = models.resnet18(pretrained=True)  # 迁移学习
        model_conv.avgpool = nn.AdaptiveAvgPool2d(1)  # 修改resnet中的平均池化层参数
        model_conv = nn.Sequential(*list(model_conv.children())[:-1])
        self.cnn = model_conv

        self.fc = nn.Linear(512, 4)

    def forward(self, img):  # 应该是重构了这个函数
        feat = self.cnn(img)
        # reshape到一维
        feat = feat.view(feat.shape[0], -1)
        c = self.fc(feat)

        return c
