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
def train(model, opt, train_loader, valid_loader):
    # 开启0号GPU训练
    use_gpu = True
    paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')
    print('start training ... ')
    model.train()
    for epoch in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            img = data[0]
            label = data[1]
            # 计算模型输出
            logits = model(img)
            # 计算损失函数
            loss_func = paddle.nn.CrossEntropyLoss(reduction='none')
            loss = loss_func(logits, label)
            # 二维变为一维：[[2.67102623]]->[2.67102623]
            avg_loss = paddle.mean(loss)
            if batch_id % 2000 == 0:
                print("epoch: {}, batch_id: {}, loss is: {:.4f}".format(epoch, batch_id, float(avg_loss.numpy())))
            avg_loss.backward()
            opt.step()
            opt.clear_grad()

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
        model.train()

    # 保存模型参数
    paddle.save(model.state_dict(), './model/mnist.pdparams')


# 创建模型
model = LeNet(num_classes=10)
# 设置迭代轮数
EPOCH_NUM = 10
# 设置优化器为Momentum，学习率为0.001
opt = paddle.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameters=model.parameters())
# 定义数据读取器
train_loader = paddle.io.DataLoader(MNIST(mode='train', transform=ToTensor()), batch_size=10, shuffle=True)
valid_loader = paddle.io.DataLoader(MNIST(mode='test', transform=ToTensor()), batch_size=1)
# 启动训练过程
train(model, opt, train_loader, valid_loader)