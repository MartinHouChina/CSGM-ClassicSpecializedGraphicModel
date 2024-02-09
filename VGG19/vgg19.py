import torch
from torch import nn
from time import time_ns


def get_tensor_bytes(x: torch.Tensor) -> int:
    return x.numel() * x.element_size()


class My_VGG19(nn.Module):
    # 把最后输出的类个数’5‘（也可为其他数字，根据数据集类别确定）改为一个可控制的变量
    def __init__(self, num_classes=4, init_weight=True):
        super(My_VGG19, self).__init__()
        self.features1 = nn.Sequential(
            # 特征提取层
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.features2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.features3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.features4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.features5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            # 分类层
            nn.Linear(in_features=7 * 7 * 512, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=num_classes)
        )

        # 参数初始化
        if init_weight:  # 如果为True，就会执行参数初始化的操作
            for m in self.modules():  # 这是一个PyTorch中的方法，用于迭代模型的所有模块。这里对模型的每一层进行遍历
                if isinstance(m, nn.Conv2d):  # 如果是卷积层，就使用kaiming初始化卷积层的权重
                    # 使用kaiming初始化，该方法适用于ReLU激活函数。它根据fan_out的方式初始化权重
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    # 如果bias不为空，固定为0
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):  # 如果是线性层
                    # 正态初始化，使用正态分布初始化权重，均值为0，标准差为0.01
                    nn.init.normal_(m.weight, 0, 0.01)
                    # bias则固定为0
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = self.features3(x)
        x = self.features4(x)
        x = self.features5(x)
        x = torch.flatten(x, 1)  # ‘1’表示从第二个维度开始展平，保留第一个维度（通常是批量大小）
        result = self.classifier(x)
        return result
