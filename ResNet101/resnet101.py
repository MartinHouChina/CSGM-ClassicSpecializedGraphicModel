import torch
from torch import nn
from time import time_ns


def get_tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


# ResNet
# 创建block块
class My_Res_Block(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        '''
        :param in_planes: 输入通道数
        :param out_planes:  输出通道数
        :param stride:  步长，默认为1
        :param downsample: 是否下采样，主要是为了res+x中两者大小一样，可以正常相加
        '''
        super(My_Res_Block, self).__init__()
        self.model = nn.Sequential(
            # 第一层是1*1卷积层：只改变通道数，不改变大小
            nn.Conv2d(in_planes, out_planes, kernel_size=1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(),
            # nn.Dropout(0.2),
            # 第二层为3*3卷积层，根据上图的介绍，可以看出输入和输出通道数是相同的
            nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(),
            # nn.Dropout(0.2),
            # 第三层1*1卷积层，输出通道数扩大四倍（上图中由64->256）
            nn.Conv2d(out_planes, out_planes * 4, kernel_size=1),
            nn.BatchNorm2d(out_planes * 4),
            # nn.ReLU(),
        )
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        res = x
        result = self.model(x)
        # 是否需要下采样来保证res与result可以正常相加
        if self.downsample is not None:
            res = self.downsample(x)
        # 残差相加
        result += res
        # 最后还有一步relu
        result = self.relu(result)
        return result


# 创建ResNet模型
class resnet101(nn.Module):
    def __init__(self, layers=101, num_classes=4, in_planes=64):
        '''
        :param layers:  我们ResNet的层数，比如常见的50、101等
        :param num_classes:  最后输出的类别数，就是softmax层的输出数目
        :param in_planes: 我们的block第一个卷积层使用的通道个数
        '''
        super(resnet101, self).__init__()
        # 定义一个字典，来存储不同resnet对应的block的个数
        # 在官方实现中，使用另外一个参数来接收，这里参考博客，采用一个字典来接收，都类似
        self.layers_dict = {
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
        }
        self.in_planes = in_planes
        # 最开始的一层，还没有进入block
        # 输入彩色，通道为3；输出为指定的
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU()
        # 根据网络结构要求，大小变为一半
        self.maxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 进入block层
        self.block1 = self._make_layer(self.layers_dict[layers][0], stride=1, planes=64)
        self.block2 = self._make_layer(self.layers_dict[layers][1], stride=2, planes=128)
        self.block3 = self._make_layer(self.layers_dict[layers][2], stride=2, planes=256)
        self.block4 = self._make_layer(self.layers_dict[layers][3], stride=2, planes=512)
        # 要经历一个平均池化层
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # 最后接上一个全连接输出层
        self.fc = nn.Linear(512 * 4, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.partitions = nn.Sequential(
            nn.Sequential(
                self.conv1,
                self.bn1,
                self.relu
            ),
            self.maxPool,
            self.block1,
            self.block2,
            self.block3,
            self.block4,
            self.avgpool,
        )

    def __getitem__(self, item):
        return self.partitions[item]

    def _make_layer(self, layers, stride, planes):
        '''
        :param planes: 最开始卷积核使用的通道数
        :param stride: 步长
        :param layers:该层 block 有多少个重复的
        :return:
        '''
        downsample = None
        # 判断是否需要下采样
        if stride != 1 or self.in_planes != planes * 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * 4),
            )
        temp_layers = []
        # 创建第一个block，第一个参数为输入的通道数，第二个参数为第一个卷积核的通道数
        temp_layers.append(My_Res_Block(self.in_planes, planes, stride, downsample))
        # 输出扩大4倍
        self.in_planes = planes * 4
        # 对于18，34层的网络，经过第一个残差块后，输出的特征矩阵通道数与第一层的卷积层个数一样
        # 对于50，101，152层的网络，经过第一个残差块后，输出的特征矩阵通道数时第一个卷积层的4倍,因此要将后续残差块的输入特征矩阵通道数调整过来
        for i in range(1, layers):
            # 输入*4，输出变为最初的
            temp_layers.append(My_Res_Block(self.in_planes, planes))
        return nn.Sequential(*temp_layers)  # 将列表解码

    def forward(self, x):
        time0 = time_ns()
        mem0 = get_tensor_bytes(x)
        # conv1_x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        time1 = time_ns()
        mem1 = get_tensor_bytes(x)
        # conv2_x
        x = self.maxPool(x)
        x = self.block1(x)

        time2 = time_ns()
        mem2 = get_tensor_bytes(x)
        # conv3_x
        x = self.block2(x)

        time3 = time_ns()
        mem3 = get_tensor_bytes(x)
        # conv4_x
        x = self.block3(x)

        time4 = time_ns()
        mem4 = get_tensor_bytes(x)
        # conv5_x
        x = self.block4(x)

        time5 = time_ns()
        mem5 = get_tensor_bytes(x)
        # average pool and fc
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        time6 = time_ns()
        mem6 = get_tensor_bytes(x)

        dfn = ("0us",
               str((time1 - time0) * (10 ** -6)) + "ms",
               str((time2 - time0) * (10 ** -6)) + "ms",
               str((time3 - time0) * (10 ** -6)) + "ms",
               str((time4 - time0) * (10 ** -6)) + "ms",
               str((time5 - time0) * (10 ** -6)) + "ms",
               str((time6 - time0) * (10 ** -6)) + "ms"
               )

        mem_seq = (
            mem0,
            mem1,
            mem2,
            mem3,
            mem4,
            mem5,
            mem6
        )

        return x, dfn, mem_seq
