# -*- coding: utf-8 -*-
# LeNet 识别手写数字
import os
import random
import paddle
import numpy as np
import paddle
from paddle.vision.transforms import ToTensor
from paddle.vision.datasets import MNIST
from LeNet import LeNet
import paddle.nn.functional as F


# 定义训练过程
def predict(model, valid_loader):
    # 开启0号GPU训练
    use_gpu = True
    paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')
    print('start predicting ... ')
    model.eval()
    accuracies = []
    losses = []
    for batch_id, data in enumerate(valid_loader()):
        img = data[0]
        label = data[1]
        # 计算模型输出
        logits = model(img)
        pred = F.softmax(logits)
        # 计算损失函数
        loss_func = paddle.nn.CrossEntropyLoss(reduction='none')
        loss = loss_func(logits, label)
        # pred为一组预测结果->一维，label为真实标签->零维
        acc = paddle.metric.accuracy(pred, label)
        accuracies.append(acc.numpy())
        losses.append(loss.numpy())
    print("[validation] accuracy/loss: {:.4f}/{:.4f}".format(np.mean(accuracies), np.mean(losses)))


# 创建模型
model = LeNet(num_classes=10)
try:
    model.load_dict(paddle.load("mnist.pdparams"))
    print("model has loaded")
except:
    raise ("no model was found")


# 定义数据读取器
valid_loader = paddle.io.DataLoader(MNIST(mode='test', transform=ToTensor()), batch_size=10)
# 启动训练过程
predict(model, valid_loader)