#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@File   :  fixed_length_baseline -> length_recognition.py
@Date   :  2021/1/20 13:50
@Author :  HJT
'''
import glob, json
import torch
from ModelClass import *
from torchvision import transforms

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
train_path = glob.glob('F:/Tianchi_OCR_Freshman/Dataset/mchar_train/*.png')
train_path.sort()  # 按照文件名排序
train_json = json.load(open('F:/Tianchi_OCR_Freshman/Dataset/mchar_train.json'))  # 是排好序了的
train_length = [len(train_json[x]['label'])-1 for x in train_json]
del_idx = preprocess(train_json)
train_path = [train_path[i] for i in range(0,len(train_path)) if i not in del_idx]
train_length = [train_length[i] for i in range(0,len(train_length)) if i not in del_idx]
trainlength_loader = torch.utils.data.DataLoader(
    LengthDataset(train_path, train_length,
                   transforms.Compose([
                       # 缩放到固定尺寸
                       transforms.Resize((64, 128)),
                       # 随机颜色变换
                       transforms.ColorJitter(0.3, 0.3, 0.3),  # 改变图像的属性，亮度、对比度、饱和度和色调
                       # 加入随机旋转
                       transforms.RandomRotation(15),  # 随机旋转，在-5~5°之间旋转
                       # 将图片转换为pytorch的tensor
                       transforms.ToTensor(),
                       transforms.Normalize([0.4442,0.437,0.4326],[0.2035,0.2060,0.2076]),   # 均值、方差
                   ])),
    batch_size=256,  # 每批样本个数
    shuffle=False,   # 是否打乱顺序
    num_workers=0,   # 读取的线程个数
)

######################  验证集构造  #######################
valid_path = glob.glob('F:/Tianchi_OCR_Freshman/Dataset/mchar_val/*.png')
valid_path.sort()
valid_json = json.load(open('F:/Tianchi_OCR_Freshman/Dataset/mchar_val.json'))  # 是排好序了的
valid_length = [len(valid_json[x]['label'])-1 for x in valid_json]
del_idx = preprocess(valid_json)
valid_path = [valid_path[i] for i in range(0,len(valid_path)) if i not in del_idx]
valid_length = [valid_length[i] for i in range(0,len(valid_length)) if i not in del_idx]
validlength_loader = torch.utils.data.DataLoader(
    LengthDataset(valid_path,valid_length,transforms.Compose([
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
    shuffle=False,  # 是否打乱顺序
    num_workers=0,  # 10, # 读取的线程个数
)

################################  函数定义区  ######################################
def length_train(train_loader, model, criterion, optimizer, epoch):
    # 训练模式
    model.train()  # 启用batch normalization和dropout
    data_num = 0
    for x in train_loader:
        for i, (input, target) in enumerate(x):
            data_num += 1
            c = model(input)  # 调用的是模型的forward函数

            loss = criterion(c, target)
            # if not (data_num % 300):
            print('Epoch: {}  Loss: {}/{}'.format(epoch + 1, loss, data_num))
            optimizer.zero_grad()  # 梯度清零(每个batch都要清零)
            loss.backward()  # 误差反向传播
            optimizer.step()  # 模型参数更新

def length_validate(valid_loader, model, criterion):
    # 测试模式
    model.eval()   # 不启用BatchNormalization和Dropout
    val_loss = []
    sum = 0

    # 不记录模型梯度信息
    with torch.no_grad():
        for i, (input, target) in enumerate(valid_loader):
            c = model(input)  # 调用的是模型的forward函数
            idx = torch.argmax(c, dim=1)  # 如果最大为10，选用第二大的
            if idx.item() == target.item():
                sum += 1
            loss = criterion(c, target)
            val_loss.append(loss.item())

    return sum*1.0/10000, np.mean(val_loss)  # 返回验证集中的平均误差值

'''  字符长度识别部分  '''
model = Length_Recognition_model()  # 基于Resnet的预训练模型
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 0.001)  # 学习率可能大了
#####################
train_epoch = 10
best_loss = 1000
for epoch in range(train_epoch):
    loader = [trainlength_loader, trainlength_loader, trainlength_loader]
    length_train(loader, model, criterion, optimizer, epoch)
    acc_loss, val_loss = length_validate(validlength_loader, model, criterion)
    print('Valid----Epoch: {}/{}  AccLoss: {}, LogLoss: {}'.format(epoch + 1, train_epoch, acc_loss, val_loss))
    # 记录下验证集精度,储存验证集误差最小时模型的参数
    if val_loss < best_loss:
        print('当前验证集最大准确率为: ', val_loss)
        best_loss = val_loss
        # state_dict()表示模型的参数字典,包含优化器对象、存储了优化器的状态、所使用到的超参数
        torch.save(model.state_dict(), './Length_model.pt')
print('长度识别模型: ', best_loss)