import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

import config
from alex_model import MultiModalModel
from config import device
from dataset import MultiModalDataset


def custom_collate_fn(batch):
    """
    自定义collate函数，用于处理数据集中的数据项并将其合并成一个批次。

    参数:
    - batch (list): 包含多个数据项的列表，每个数据项可以是图像、文本输入和标签的元组，或者只是图像和文本输入的元组。

    返回:
    - images (Tensor): 一个包含所有图像的张量。
    - text_inputs (dict): 包含所有文本输入的字典，每个文本输入都由input_ids、attention_mask和token_type_ids组成。
    - labels (Tensor, 可选): 如果数据项中包含标签，则返回一个包含所有标签的张量。
    """
    # 初始化图像、文本输入和标签的列表
    images = []
    text_inputs = []
    labels = []

    # 遍历批次中的每个数据项
    for item in batch:
        # 检查数据项的长度以确定是否包含标签
        if len(item) == 3:
            # 如果包含标签，将数据项解包为图像、文本输入和标签
            image, text_input, label = item
            # 将标签添加到标签列表中
            labels.append(label)
        else:
            # 如果不包含标签，将数据项解包为图像和文本输入
            image, text_input = item
        # 将图像和文本输入分别添加到相应的列表中
        images.append(image)
        text_inputs.append(text_input)

    # 将图像列表堆叠成一个张量
    images = torch.stack(images, dim=0)

    # 从文本输入列表中提取input_ids、attention_mask和token_type_ids，并进行填充
    input_ids = pad_sequence([item['input_ids'] for item in text_inputs], batch_first=True)
    attention_mask = pad_sequence([item['attention_mask'] for item in text_inputs], batch_first=True)
    token_type_ids = pad_sequence([item['token_type_ids'] for item in text_inputs], batch_first=True)

    # 将提取的文本输入张量组合成一个字典
    text_inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids
    }

    # 如果标签列表不为空，将其转换为张量并返回图像、文本输入和标签
    if labels:
        labels = torch.tensor(labels, dtype=torch.long)
        return images, text_inputs, labels
    else:
        # 如果标签列表为空，仅返回图像和文本输入
        return images, text_inputs


# 定义训练模型的函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=config.epochs,
                model_save_path='best_model.pth'):
    """
    训练模型并保存最佳模型。

    参数:
    model: 要训练的模型。
    train_loader: 训练数据加载器。
    val_loader: 验证数据加载器。
    criterion: 损失函数。
    optimizer: 优化器。
    num_epochs: 训练的轮数，默认为3。
    model_save_path: 最佳模型保存路径，默认为'best_model.pth'。
    """
    # 初始化最佳准确率为0.0
    best_accuracy = 0.0
    # 遍历每个epoch
    for epoch in range(num_epochs):
        model.train()
        # 初始化当前epoch的累计损失为0.0
        running_loss = 0.0
        # 遍历训练数据加载器中的每个批次
        for images, text_inputs, labels in train_loader:
            # 将数据移动到GPU
            images = images.to(device)
            for key in text_inputs:
                text_inputs[key] = text_inputs[key].to(device)
            labels = labels.to(device)

            # 清零优化器的梯度
            optimizer.zero_grad()
            # 前向传播
            outputs = model(images, text_inputs)
            # 计算损失
            loss = criterion(outputs, labels)
            # 反向传播
            loss.backward()
            # 更新模型参数
            optimizer.step()
            # 累加损失
            running_loss += loss.item()
        # 打印当前epoch的损失
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

        model.eval()
        # 初始化正确预测数和总预测数为0
        correct = 0
        total = 0
        # 在无梯度计算的情况下进行验证
        with torch.no_grad():
            # 遍历验证数据加载器中的每个批次
            for images, text_inputs, labels in val_loader:
                # 将数据移动到 GPU
                images = images.to(device)
                for key in text_inputs:
                    text_inputs[key] = text_inputs[key].to(device)
                labels = labels.to(device)

                # 前向传播
                outputs = model(images, text_inputs)
                # 获取预测结果
                _, predicted = torch.max(outputs.data, 1)
                # 累加总预测数
                total += labels.size(0)
                # 累加正确预测数
                correct += (predicted == labels).sum().item()
        # 计算准确率
        accuracy = 100 * correct / total
        # 打印验证准确率
        print(f"Validation Accuracy: {accuracy}%")

        # 如果当前准确率高于最佳准确率，则保存模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at {model_save_path} with accuracy: {best_accuracy}%")


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
    train_dataset = MultiModalDataset(data_dir='data', samples=train_samples, transform=config.transform)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=custom_collate_fn)

    val_dataset = MultiModalDataset(data_dir='data', samples=val_samples, transform=config.transform)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=custom_collate_fn)

    # 模型、损失函数和优化器
    model = MultiModalModel().to(device)  # 将模型移动到 GPU
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learn_rate, weight_decay=config.weight_decay)

    train_model(model, train_loader, val_loader, criterion, optimizer)
