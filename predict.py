# 预测测试集
import pandas as pd
import torch
from torch.utils.data import DataLoader

import config
from config import device,transform
from dataset import MultiModalDataset
from train import custom_collate_fn
from Resnet50_model import MultiModalModel


def predict(model, test_loader, device):
    """
    使用训练好的模型进行预测。

    参数:
    model: 训练好的模型。
    test_loader: 测试数据加载器，用于迭代加载测试数据。
    device: 设备信息，用于确定使用CPU还是GPU进行计算。

    返回:
    text_predictions: 预测的文本标签列表。
    """
    # 将模型设置为评估模式
    model.eval()
    # 初始化预测结果列表
    predictions = []

    # 在预测过程中不需要计算梯度，因此使用torch.no_grad()来禁用梯度计算，以减少内存消耗
    with torch.no_grad():
        # 迭代加载测试数据
        for images, text_inputs in test_loader:
            # 将数据移动到指定设备
            images = images.to(device)
            for key in text_inputs:
                text_inputs[key] = text_inputs[key].to(device)

            # 使用模型进行预测
            outputs = model(images, text_inputs)
            # 获取最高分数对应的类别
            _, predicted = torch.max(outputs.data, 1)
            # 将预测结果添加到列表中
            predictions.extend(predicted.tolist())

    # 获取将整数标签转换回文本标签的逆映射字典
    int_to_label = test_dataset.int_to_label
    # 将预测的整数标签转换为文本标签
    text_predictions = [int_to_label[pred] for pred in predictions]
    # 返回预测的文本标签列表
    return text_predictions


if __name__ == '__main__':
    # 加载模型结构
    model = MultiModalModel().to(device)  # 将模型移动到 GPU

    # 加载保存的权重
    model.load_state_dict(torch.load('best_model.pth', weights_only=True))

    # 加载需要预测的数据
    test_data = pd.read_csv('test_without_label.txt', header=0, names=['guid', 'tag'])
    test_dataset = MultiModalDataset(data_dir='data', samples=test_data, transform=transform, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=custom_collate_fn)

    # 执行预测
    text_predictions = predict(model, test_loader, device)
    # 保存预测结果
    test_data['tag'] = text_predictions  # 使用文本标签而不是整数标签
    test_data.to_csv('test_without_label.txt', index=False)
