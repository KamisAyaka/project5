import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.models.resnet import ResNet50_Weights

import config
from config import device


class ImageDataset(Dataset):
    """
    自定义图像数据集类，继承自torch.utils.data.Dataset。
    用于加载和处理图像数据，以及对应的标签。

    参数:
    - data_dir (str): 数据集目录路径，包含所有图像。
    - samples (pd.DataFrame): 包含样本信息的DataFrame，第一列是样本ID，第二列是标签。
    - transform (callable, optional): 可选的图像变换函数，用于图像预处理。
    """

    def __init__(self, data_dir, samples, transform=None):
        """
        初始化ImageDataset类。

        将数据集目录、样本信息和图像变换（如果提供）保存为类的属性。
        同时，定义一个标签到整数的映射，用于后续的标签编码。
        """
        self.data_dir = data_dir
        self.samples = samples
        self.transform = transform
        # 标签到整数的映射，将分类标签转换为数字
        self.label_to_int = {'negative': 0, 'neutral': 1, 'positive': 2}

    def __len__(self):
        """
        实现len方法，返回数据集中样本的数量。

        返回:
        - int: 样本数量。
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        实现索引方法，根据索引idx返回对应的图像和标签。

        参数:
        - idx (int): 样本索引。

        返回:
        - tuple: 包含图像和对应标签的元组。
        """
        # 获取样本的唯一标识符和标签
        guid = self.samples.iloc[idx, 0]
        label = self.samples.iloc[idx, 1]

        # 将字符串标签转换为整数
        label = self.label_to_int[label]

        # 构造图像路径并加载图像
        image_path = os.path.join(self.data_dir, f"{guid}.jpg")
        image = Image.open(image_path).convert('RGB')

        # 如果提供了变换函数，则对图像进行变换
        if self.transform:
            image = self.transform(image)

        # 将标签转换为torch tensor
        label = torch.tensor(label, dtype=torch.long)

        # 返回图像和标签
        return image, label


def custom_collate_fn(batch):
    """
    自定义的collate函数用于处理数据集中的数据项。
    PyTorch的DataLoader使用此函数来组合多个样本到一个批次。

    参数:
    batch (list): 一个批次的数据项列表，每个数据项是一个图像-标签对。

    返回:
    tuple: 一个包含图像批次和标签批次的元组。
    """

    # 初始化图像和标签的列表
    images = []
    labels = []

    # 遍历批次中的每个数据项
    for item in batch:
        # 将数据项分解为图像和标签
        image, label = item
        # 将标签添加到标签列表中
        labels.append(label)
        # 将图像添加到图像列表中
        images.append(image)

    # 将图像列表堆叠为一个张量
    images = torch.stack(images, dim=0)

    # 将标签列表转换为张量
    labels = torch.tensor(labels, dtype=torch.long)

    # 返回图像和标签的张量
    return images, labels


# 定义训练模型的函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=config.epochs):
    """
    参数:
      model: 要训练的模型
      train_loader: 训练数据加载器
      val_loader: 验证数据加载器
      criterion: 损失函数
      optimizer: 优化器
      num_epochs: 训练的轮数，默认为 config.epochs
    """
    # 训练循环
    for epoch in range(num_epochs):
        model.train()  # 将模型设置为训练模式
        running_loss = 0.0  # 初始化累计损失
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()  # 清零梯度
            outputs = model(images)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重
            running_loss += loss.item()  # 累计损失
        # 打印当前轮次的平均损失
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

        model.eval()  # 将模型设置为评估模式
        correct = 0  # 初始化正确预测数
        total = 0  # 初始化总预测数
        with torch.no_grad():  # 不计算梯度
            for images, labels in val_loader:
                images = images.to(device)  # 确保 images 在 config.device 上
                labels = labels.to(device)  # 确保 labels 在 config.device 上

                outputs = model(images)  # 前向传播
                _, predicted = torch.max(outputs.data, 1)  # 获取预测标签
                total += labels.size(0)  # 累计总样本数
                correct += (predicted == labels).sum().item()  # 累计正确预测数
        accuracy = 100 * correct / total  # 计算准确率
        # 打印验证集上的准确率
        print(f"Validation Accuracy: {accuracy}%")



# 主函数
if __name__ == '__main__':
    # 读取 train.txt 文件
    train_data = pd.read_csv('train.txt', header=0, names=['guid', 'tag'])

    # 随机打乱数据
    train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)

    # 分割数据
    split_ratio = 0.8
    train_size = int(len(train_data) * split_ratio)
    train_samples = train_data[:train_size]
    val_samples = train_data[train_size:]

    # 创建数据集和数据加载器
    train_dataset = ImageDataset(data_dir='data', samples=train_samples, transform=config.transform)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=custom_collate_fn)

    val_dataset = ImageDataset(data_dir='data', samples=val_samples, transform=config.transform)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=custom_collate_fn)

    # 模型、损失函数和优化器
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learn_rate, weight_decay=config.weight_decay)

    train_model(model, train_loader, val_loader, criterion, optimizer)
