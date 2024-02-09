import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from dataset import My_Dataset_test
from ResNet101.resnet101 import resnet101


def validate_model(model):
    # 批量数目
    batch_size = 40
    # 预测正确个数
    correct = 0
    # 将模型放入GPU中
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # 加载数据
    test_set = My_Dataset_test('../data/RSI-CB256_split/val', transform=transforms.ToTensor())
    test_loader = DataLoader(test_set, batch_size, shuffle=True)
    # 开始
    for batch_data, batch_label in test_loader:
        # 放入GPU中
        batch_data, batch_label = batch_data.cuda(), batch_label.cuda()
        # 预测
        prediction = model(batch_data)
        # 将预测值中最大的索引取出，其对应了不同类别值
        predicted = torch.max(prediction.data, 1)[1]
        # 获取准确个数
        correct += (predicted == batch_label).sum()
    print(test_set.__len__())
    print('测试准确率: %.2f %%' % (100 * correct / test_set.__len__()))  # 因为总共(500)个测试数据


if __name__ == '__main__':
    model = resnet101
    print(model[2])
