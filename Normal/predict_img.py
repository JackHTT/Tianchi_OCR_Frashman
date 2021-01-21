#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@File   :  定长字符识别baseline -> predict_img.py
@Date   :  2021/1/18 10:09
@Author :  HJT
'''

import os, glob, json
import torch
from torch import nn
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import pandas as pd
import torch.nn.functional as F
from ModelClass import *

'''
# 模型预测部分及准确率指标
'''
model_path = 'ResNet_model.pt'
model = SVHN_Resnet_model()
model.load_state_dict(torch.load(model_path))
model.eval()

model_path2 = 'Length_model.pt'
model2 = Length_Recognition_model()
model2.load_state_dict(torch.load(model_path2))
model2.eval()

# 看下验证集的准确率
test_path = glob.glob('F:/Tianchi_OCR_Freshman/Dataset/mchar_test_a/*.png')
test_path.sort()
img_name = os.listdir('F:/Tianchi_OCR_Freshman/Dataset/mchar_test_a/')
img_name.sort()

test_loader = torch.utils.data.DataLoader(
    SVHNDataset(test_path,[],transforms.Compose([
        # 缩放到固定尺寸
        transforms.Resize((64, 128)),
        # 随机颜色变换
        # transforms.ColorJitter(0.2, 0.2, 0.2),  # 改变图像的属性，亮度、对比度、饱和度和色调
        # 加入随机旋转
        # transforms.RandomRotation(5),  # 随机旋转，在-5~5°之间旋转
        transforms.ToTensor(),
        transforms.Normalize([0.4452,0.4363,0.4266],[0.1989,0.1992,0.2029]),   # 均值、方差

    ])),
    batch_size=1,  # 每批样本个数  # 一个batch的样本数目通常设为2的n次幂，网络较小时选用256，较大时选用64
    shuffle=False,  # 是否打乱顺序
    num_workers=0,  # 10, # 读取的线程个数
)

testaug_loader = torch.utils.data.DataLoader(
    SVHNDataset(test_path,[],transforms.Compose([
        # 缩放到固定尺寸
        transforms.Resize((64, 128)),
        # 随机颜色变换
        # transforms.ColorJitter(0.2, 0.2, 0.2),  # 改变图像的属性，亮度、对比度、饱和度和色调
        # 加入随机旋转
        # transforms.RandomRotation(5),  # 随机旋转，在-5~5°之间旋转
        transforms.ToTensor(),
        transforms.Normalize([0.4452,0.4363,0.4266],[0.1989,0.1992,0.2029]),   # 均值、方差

    ])),
    batch_size=1,  # 每批样本个数  # 一个batch的样本数目通常设为2的n次幂，网络较小时选用256，较大时选用64
    shuffle=False,  # 是否打乱顺序
    num_workers=0,  # 10, # 读取的线程个数
)

total_sum = 0
result = pd.DataFrame()
predict_label = []
predict_seq1, predict_seq2 = [], []
predict_seq_sum = []
predict_len = []

result['file_name'] = pd.Series(img_name)

def convert_num(c):
    idx = [torch.argmax(c[j], dim=1).item() for j in range(4)]  # 如果最大为10，选用第二大的
    # print(idx)
    he = 0
    for k in range(0,len(idx)):
        if idx[k] == 10:
            break
        he = he * 10 + idx[k]
    return he

def convert_to_num(num):  # num为1*44的list
    sum = 0
    for i in range(4):
        aa = list(num[0+i*11 : 11+i*11])
        idx = aa.index(max(aa))
        if idx == 10:
            break
        sum = sum * 10 + idx
    return sum

def convert_to_num2(num, length):
    # 假设长度识别的很准
    index_list = []
    count = 0
    for i in range(4):
        aa = list(num[0+i*11 : 11+i*11])
        if i < length:
            idx = aa.index(max(aa))
            if idx == 10:
                del aa[10]
                idx = aa.index(max(aa))
        else:
            idx = aa.index(max(aa))
        if int(idx) != 10:
            count += 1
            index_list.append(idx)
    sum = 0
    for i in range(count):
        sum = sum * 10 + index_list[i]

    return sum

#####################################################
print('start predict length!!!')
for i, (input, target) in enumerate(test_loader):
    if i % 400 == 0:
        print(i)

    len = torch.argmax(model2(input), dim=1).item()
    predict_len.append(len+1)
    if i > 60:
        break

print('start predict number!!!')
def predict(test_loader, model, tta = 10):
    model.eval()
    test_pred_tta = None
    # TTA 次数
    for _ in range(tta):
        test_pred = []

        with torch.no_grad():
            for i, (input, target) in enumerate(test_loader):
                if i % 400 == 0:
                    print(_,':  ',i)
                c = model(input)
                # 1 * 44
                op = np.concatenate([c[0].data.numpy(), c[1].data.numpy(),
                                         c[2].data.numpy(), c[3].data.numpy()], axis=1)
                # print(i,':  ',output)
                # break

                test_pred.append(op)
                if i > 60:
                    break

        test_pred = np.vstack(test_pred)
        if test_pred_tta is None:
            test_pred_tta = test_pred
        else:
            test_pred_tta += test_pred

    return test_pred_tta

output = predict(test_loader, model, 1)
# print(output)
# print(output[0])
# print(output)
# print(predict_len)
pred_num = []
# print(output[0])
# print(len(output))
for i in range(62):
    pred_num.append(convert_to_num2(output[i], predict_len[i]))
result['file_code'] = pd.Series(pred_num)
print(predict_len)
print(pred_num)

# print(pred_num)
# print(type(output))
# print(output)
'''
for i, (input, target) in enumerate(test_loader):
    if i % 400 == 0:
        print(i)
    c = model(input)  # 调用的是模型的forward函数
    # predict = [c[x].detach().numpy().tolist() for x in range(4)]
    # predict = [F.softmax(c[x],dim=1).detach().numpy().tolist() for x in range(4)]
    # predict_seq1.append(predict[0])
    predict_label.append(convert_num(c))
result['file_code'] = pd.Series(predict_label)
'''

# predict_seq = []
# for i in range(len(predict_seq1)):
#     l = []
#     for j in range(4):
#         l1 = []
#         for k in range(11):
#             l1.append((predict_seq1[i][j][k] + predict_seq2[i][j][k])/2)
#         l.append(l1)
#     predict_seq.append(l)
#     '''
#     output = model(input)  # 调用的是模型的forward函数
#     # predict = [c[x].detach().numpy().tolist() for x in range(4)]
#     predict = [F.softmax(output[x], dim=1).detach().numpy().tolist() for x in range(4)]
#     predict_seq.append(predict)
#     predict_label.append(convert_num(output))
#     '''
#
# result['pred_len'] = pd.Series(predict_len)
# result['pred_num'] = pd.Series(predict_label)
# result['pred_seq1'] = pd.Series(predict_seq1)
# result['pred_seq2'] = pd.Series(predict_seq2)
# result['pred_seq'] = pd.Series(output)

'''
# TTA

for x in [test_loader, testaug_loader]:
    for i, (input, target) in enumerate(x):
        # total_sum += 1
        if i % 400 == 0:
            print(i)
        c = model(input)  # 调用的是模型的forward函数
        # predict = [c[x].detach().numpy().tolist() for x in range(4)]
        predict = [F.softmax(c[x],dim=1).detach().numpy().tolist() for x in range(4)]
        predict_seq.append(predict)

    predict_seq_sum.append(predict_seq)
    predict_seq = []

# predict_seq_ave = (predict_seq_sum[0] + predict_seq_sum[1])/2

# result['file_code'] = pd.Series(predict_label)
result['file_seq1'] = pd.Series(predict_seq_sum[0])
result['file_seq2'] = pd.Series(predict_seq_sum[1])
'''

result_path = 'F:/Tianchi_OCR_Freshman/reslut/'
result.to_csv(result_path+'submission1.csv',index=False)






