# 训练过程
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models import resnet101
from torchvision.transforms import transforms
from dataset import My_Dataset
from validation import validate_model
import matplotlib.pyplot as plt
import numpy as np

# 模型保存路径(后缀是.pth):
model_saved_path = ""

# 训练集路径:
dataset_path = ""


def train():
    batch_size = 40  # 批量训练大小

    from ResNet101 import resnet101
    model = resnet101()  # 传入模型，例如 resnet101()

    # 将模型放入GPU中
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # 定义损失函数
    loss_func = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = optim.SGD(params=model.parameters(), lr=0.003)
    # 加载数据
    train_set = My_Dataset('../data/RSI-CB256_split/train', transform=transforms.ToTensor())
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    loss_list = []
    correct_list = []
    # 训练
    for i in range(30):
        correct = 0  # 正确个数
        loss_temp = 0  # 临时变量
        for j, (batch_data, batch_label) in enumerate(train_loader):
            # 数据放入GPU中
            batch_data, batch_label = batch_data.cuda(), batch_label.cuda()
            # 梯度清零
            optimizer.zero_grad()
            # 模型训练
            prediction = model(batch_data)
            # 将预测值中最大的索引取出，其对应了不同类别值
            predicted = torch.max(prediction.data, 1)[1]
            # 获取准确个数
            correct += (predicted.cpu() == batch_label.cpu()).sum()
            # 损失值
            loss = loss_func(prediction, batch_label)
            loss_temp += loss.item()
            # 反向传播
            loss.backward()
            # 梯度更新
            optimizer.step()

        correct_list.append(correct / train_set.__len__())
        # 打印一次损失值
        print('[%d] loss: %.3f' % (i + 1, loss_temp / len(train_loader)))
        # 打印一次准确率
        print('训练准确率: %.2f %%' % (100 * correct / train_set.__len__()))  # 训练集共(1000)个数据
        loss_list.append(loss_temp / len(train_loader))
    torch.save(model, model_saved_path)
    validate_model(model)
    print(correct_list)
    episodes_list = list(range(len(loss_list)))
    plt.xticks(np.arange(0, len(episodes_list), len(episodes_list) // 10))
    plt.xlim(0, len(episodes_list))
    plt.plot(episodes_list, loss_list, lw=2)
    plt.plot(episodes_list, correct_list, color='red', lw=2)
    plt.xlabel('Episodes')
    plt.ylabel('loss/acc')
    plt.legend(['train loss', 'train acc'], loc='best', framealpha=1, edgecolor='black')
    plt.show()
