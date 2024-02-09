import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np


class My_Dataset(Dataset):
    def __init__(self, filename, transform=None):
        self.filename = filename  # 文件路径
        self.transform = transform  # 是否对图片进行变化
        self.image_name, self.label_image = self.operate_file()

    def __len__(self):
        return len(self.image_name)

    def __getitem__(self, idx):
        # 由路径打开图片
        image = Image.open(self.image_name[idx])
        # 下采样： 因为图片大小不同，需要下采样为224*224
        trans = transforms.RandomResizedCrop(224)
        image = trans(image)
        # 获取标签值
        label = self.label_image[idx]
        # 是否需要处理
        if self.transform:
            image = self.transform(image)
        # 转为tensor对象
        label = torch.from_numpy(np.array(label))
        return image, label

    def operate_file(self):
        # 获取所有的文件夹路径 '../data/net_train_images'的文件夹
        dir_list = os.listdir(self.filename)
        # 拼凑出图片完整路径 '../data/net_train_images' + '/' + 'xxx.jpg'
        full_path = [self.filename + '/' + name for name in dir_list]
        # 获取里面的图片名字
        name_list = []
        for i, v in enumerate(full_path):
            temp = os.listdir(v)  # 获取该文件夹下所有文件名
            temp_list = [v + '/' + j for j in temp]  # 将文件名与完整路径拼接，得到文件的完整路径列表'temp_list'
            # 将每个文件夹的完整路径列表合并到'name_list'中
            name_list.extend(temp_list)
        # 由于一个文件夹的所有标签都是同一个值，而字符值必须转为数字值，因此我们使用数字0-4代替标签值
        label_list = []
        # temp_list = np.array([0, 1, 2, 3, 4], dtype=np.int64)  # 用数字代表不同类别
        temp_list = np.arange(0, len(full_path), 1)
        # 将标签每个复制(200)个
        for i, v in enumerate(full_path):
            pic_list = os.listdir(v)
            for k in range(len(pic_list)):  # 训练集每类(200)张图片
                label_list.append(temp_list[i])
        # 所有文件的完整路径和对应的标签值
        return name_list, label_list


# 继承自训练数据加载器，只修改一点点的地方
class My_Dataset_test(My_Dataset):
    def operate_file(self):
        # 获取所有的文件夹路径
        dir_list = os.listdir(self.filename)
        full_path = [self.filename + '/' + name for name in dir_list]
        # 获取里面的图片名字
        name_list = []
        for i, v in enumerate(full_path):
            temp = os.listdir(v)
            temp_list = [v + '/' + j for j in temp]
            name_list.extend(temp_list)
        # 将标签每个复制一百个
        label_list = []
        # temp_list = np.array([0, 1, 2, 3, 4], dtype=np.int64)  # 用数字代表不同类别
        temp_list = np.arange(0, len(full_path), 1)
        for i, v in enumerate(full_path):
            pic_list = os.listdir(v)
            for k in range(len(pic_list)):  # 只修改了这里，测试集每类(100)张图片
                label_list.append(temp_list[i])
        return name_list, label_list
