#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# 数据扩增：颜色空间、尺度空间、样本空间
# 常用数据扩增库:torchvision、imgaug、albumentations

import os, glob, json
import numpy as np
import torch
from torch import nn
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from ModelClass import *
from collections import OrderedDict
from torch import optim

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

'''
# 模型训练函数
'''
def train(loader, model, criterion, optimizer, epoch, loss_plot, c0_plot):
    # 训练模式
    model.train()  # 启用batch normalization和dropout
    data_num = 0

    for x in loader:
        for i, (input, target) in enumerate(x):
            data_num += 1
            c = model(input)  # 调用的是模型的forward函数

            weight = [0.3, 0.3, 0.3, 0.1]
            loss = criterion(c[0], target[:, 0])
            for i in range(1,4):
                loss += criterion(c[i], target[:, i])

            loss /= 4
            loss_plot.append(loss.item())
            c0_plot.append((c[0].argmax(dim=1)==target[:, 0]).sum().item()*1.0/c[0].shape[0]) # 统计每个batch中第一个字符的预测准确率

            print('Epoch: {}  Loss: {}/{}'.format(epoch, loss, data_num))
            optimizer.zero_grad()  # 梯度清零(每个batch都要清零)
            loss.backward()  # 误差反向传播
            optimizer.step()  # 模型参数更新

'''
# 模型验证函数
'''
# 定义6位定长字符的准确率，以及一些策略
def accuracy_loss(predict, true):
    idx = [torch.argmax(predict[i],dim=1) for i in range(4)]  # 如果最大为10，选用第二大的
    for i in range(0,4):
        if true[:, i].item() == 10:
            return True
        if idx[i].item() != true[:, i].item():
            return False

    return True

def validate(valid_loader, model, criterion):
    # 测试模式
    model.eval()   # 不启用BatchNormalization和Dropout
    val_loss = []

    # 不记录模型梯度信息
    with torch.no_grad():
        acc_sum = 0
        acc_num = 0
        for i, (input, target) in enumerate(valid_loader):
            acc_sum += 1
            c = model(input)  # 调用的是模型的forward函数
            # acc
            if accuracy_loss(c, target):
                acc_num += 1

            # logloss
            # weight = [0.3, 0.3, 0.3, 0.1]
            loss = criterion(c[0], target[:, 0])
            for i in range(1, 4):
                loss += criterion(c[i], target[:, i])
            loss /= 4
            val_loss.append(loss.item())

    return acc_num*1.0/acc_sum, np.mean(val_loss)  # 返回验证集中的平均误差值

# 删除训练集和验证集中包含超过四个数字的图片
def preprocess(json):
    png_idx = []
    index = 0
    for i in json:
        if len(json[i]['label']) > 4:
            png_idx.append(index)
        index += 1
    return png_idx

######################  训练集构造  #######################
train_path = glob.glob('./data/mchar_train/*.png')
train_path.sort()  # 按照文件名排序
train_json = json.load(open('./data/mchar_train.json'))  # 是排好序了的
train_json = sorted(train_json.items())
train_json = OrderedDict(train_json)
train_label = [train_json[x]['label'] for x in train_json]  # 有顺序的label
del_idx = preprocess(train_json)
train_path = [train_path[i] for i in range(0,len(train_path)) if i not in del_idx]
train_label = [train_label[i] for i in range(0,len(train_label)) if i not in del_idx]

train_loader = torch.utils.data.DataLoader(
    SVHNDataset(train_path, train_label,
                   transforms.Compose([
                       # 缩放到固定尺寸
                       transforms.Resize((64, 128)),
                       # 随机颜色变换
                       transforms.ColorJitter(0.5, 0.5, 0.5),  # 改变图像的属性，亮度、对比度、饱和度和色调
                       # 加入随机旋转
                       transforms.RandomRotation(15),  # 随机旋转，在-5~5°之间旋转
                       # 将图片转换为pytorch的tensor
                       transforms.ToTensor(),
                       # 对图像像素进行归一化(按通道)
                       transforms.Normalize([0.4442,0.437,0.4326],[0.2035,0.2060,0.2076]),   # 均值、方差
                   ])),
    batch_size=256,  # 每批样本个数
    shuffle=True,   # 是否打乱顺序
    num_workers=0,   # 读取的线程个数
)

######################  验证集构造  #######################
valid_path = glob.glob('./data/mchar_val/*.png')
valid_path.sort()
valid_json = json.load(open('./data/mchar_val.json'))  # 是排好序了的
valid_json = sorted(valid_json.items())
valid_json = OrderedDict(valid_json)
valid_label = [valid_json[x]['label'] for x in valid_json]  # 有顺序的label
del_idx = preprocess(valid_json)
valid_path = [valid_path[i] for i in range(0,len(valid_path)) if i not in del_idx]
valid_label = [valid_label[i] for i in range(0,len(valid_label)) if i not in del_idx]

valid_loader = torch.utils.data.DataLoader(
    SVHNDataset(valid_path,valid_label,transforms.Compose([
        # 缩放到固定尺寸
        transforms.Resize((64, 128)),
        # 随机颜色变换
        # transforms.ColorJitter(0.2, 0.2, 0.2),  # 改变图像的属性，亮度、对比度、饱和度和色调
        # 加入随机旋转
        # transforms.RandomRotation(5),  # 随机旋转，在-5~5°之间旋转
        transforms.ToTensor(),
        transforms.Normalize([0.4264,0.4101,0.3946],[0.2236,0.2293,0.2361]),   # 均值、方差
    ])),
    batch_size=1, # 每批样本个数  # 一个batch的样本数目通常设为2的n次幂，网络较小时选用256，较大时选用64
    shuffle=True,  # 是否打乱顺序
    num_workers=0,  # 10, # 读取的线程个数
)

'''  不定长字符识别部分  '''
###############   模型定义   ################
model1 = SVHN_Resnet_model()

criterion = nn.CrossEntropyLoss()  # softmax-log-NLLLoss

lr = 0.002
optimizer = torch.optim.Adam(model1.parameters(), lr)  # 学习率可能大了
# step_size理解为经过几个epoch学习率发生一次变化，gamma表示每次变化时学习率变化的比例
scheduler_lr = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

best_loss = 1000
train_epoch = 10
loss_plot, c0_plot = [], []  # 储存训练过程的误差以及第一个字符的准确率

for epoch in range(1, train_epoch+1):
    loader = [train_loader, train_loader, train_loader]
    train(loader, model1, criterion, optimizer, epoch, loss_plot, c0_plot)
    scheduler_lr.step()
    acc_loss, val_loss = validate(valid_loader, model1, criterion)
    print('Valid----Epoch: {}/{}  AccLoss: {}  LogLoss: {}'.format(epoch, train_epoch, acc_loss, val_loss))

    if val_loss < best_loss:
        print('当前验证集最大准确率为: ', val_loss)
        best_loss = val_loss

        torch.save(model1.state_dict(), './ResNet_model1.pt')  # state_dict()表示模型的参数字典,包含优化器对象、存储了优化器的状态、所使用到的超参数
print('模型验证最大准确率: ', best_loss)

###############   可视化训练误差    #################
x = range(1,len(loss_plot)+1)
plt.figure()
plt.plot(x,loss_plot)
plt.title('Train Loss')

x = range(1,len(c0_plot)+1)
plt.figure()
plt.plot(x,c0_plot)
plt.title('c0 Accuracy')
plt.show()



