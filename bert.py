import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification

import config
from config import device


class TextDataset(Dataset):
    def __init__(self, data_dir, samples, tokenizer, max_length=config.context_max_len):
        """
        初始化文本数据集
        :param data_dir: 数据目录，包含文本数据
        :param samples: 数据样本，通常包含样本ID和标签
        :param tokenizer: BERT分词器
        :param max_length: 文本的最大长度
        """
        self.data_dir = data_dir
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 创建标签到整数的映射
        self.label_to_int = {'negative': 0, 'neutral': 1, 'positive': 2}

    def __len__(self):
        """
        获取数据集大小
        :return: 数据集样本数
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        根据索引获取数据集中的样本
        :param idx: 样本索引
        :return: 处理后的样本，包括文本和标签
        """
        guid = self.samples.iloc[idx, 0]
        label = self.samples.iloc[idx, 1]

        # 将标签转换为整数
        label = self.label_to_int[label]

        # 读取文本文件
        text_path = os.path.join(self.data_dir, f"{guid}.txt")
        with open(text_path, 'r', encoding='utf-8', errors='replace') as file:
            text = file.read().strip()

        # 使用tokenizer对输入文本进行编码，返回包含输入序列的字典
        text_input = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 移除字典中每个张量的冗余维度，确保每个张量的维度都是(1, -1)
        text_input = {key: val.squeeze(0) for key, val in text_input.items()}

        # 将标签转换为PyTorch长整型张量，并与文本输入一起返回
        label = torch.tensor(label, dtype=torch.long)
        return text_input, label


def text_only_collate_fn(batch):
    """
    自定义collate函数以处理文本输入的batch数据。

    该函数的目的是将一个列表中的字典对象（代表单个数据项）合并为一个批次张量，
    专用于文本输入的处理。它提取文本输入和标签，将它们分别合并，并转换为适当的张量格式。

    参数:
    - batch (list): 包含多个数据项的列表，每个数据项是一个元组，其中包含文本输入和对应的标签。

    返回:
    - dict: 包含两个键值对的字典，'input_ids' 和 'attention_mask' 分别对应输入文本的张量和注意力掩码张量。
    - labels: 包含批次中所有数据项标签的张量。
    """
    # 初始化文本输入和标签的列表
    text_inputs = []
    labels = []

    # 遍历批次中的每个数据项，分离文本输入和标签
    for item in batch:
        text_input, label = item
        labels.append(label)
        text_inputs.append(text_input)

    # 将文本输入列表转换为张量
    input_ids = torch.stack([item['input_ids'] for item in text_inputs], dim=0)
    attention_mask = torch.stack([item['attention_mask'] for item in text_inputs], dim=0)

    # 将标签列表转换为张量
    labels = torch.tensor(labels, dtype=torch.long)

    # 返回处理后的文本输入张量和标签张量
    return {'input_ids': input_ids, 'attention_mask': attention_mask}, labels


def train_text_only_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=config.epochs):
    """
    训练一个仅基于文本的模型，并在训练过程中验证模型性能。

    参数:
    model: 要训练的模型实例。
    train_loader: 训练数据加载器，用于迭代加载训练数据。
    val_loader: 验证数据加载器，用于迭代加载验证数据。
    criterion: 损失函数，用于计算模型输出与真实标签之间的差异。
    optimizer: 优化器，用于更新模型参数。
    num_epochs: 训练的轮数，默认值为config.epochs。
    """
    for epoch in range(num_epochs):
        # 在每个epoch开始时，将模型设置为训练模式
        model.train()
        running_loss = 0.0

        # 遍历训练数据集
        for text_inputs, labels in train_loader:
            # 将输入数据和标签移动到指定设备
            input_ids = text_inputs['input_ids'].to(device)
            attention_mask = text_inputs['attention_mask'].to(device)
            labels = labels.to(device)

            # 清除之前的梯度
            optimizer.zero_grad()
            # 前向传播
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # 计算损失
            loss = criterion(outputs.logits, labels)
            # 反向传播
            loss.backward()
            # 更新模型参数
            optimizer.step()
            # 累加损失
            running_loss += loss.item()

        # 打印当前epoch的损失
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

        # 在每个epoch结束时，将模型设置为评估模式
        model.eval()
        correct = 0
        total = 0
        # 在评估模式下，不计算梯度
        with torch.no_grad():
            # 遍历验证数据集
            for text_inputs, labels in val_loader:
                # 将输入数据和标签移动到指定设备
                input_ids = text_inputs['input_ids'].to(device)
                attention_mask = text_inputs['attention_mask'].to(device)
                labels = labels.to(device)

                # 前向传播
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                # 获取预测的标签
                _, predicted = torch.max(outputs.logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # 计算验证集上的准确率
        accuracy = 100 * correct / total
        # 打印当前epoch的验证准确率
        print(f"Validation Accuracy: {accuracy}%")


if __name__ == '__main__':
    # 读取 train.csv 文件
    train_data = pd.read_csv('train.txt', header=0, names=['guid', 'tag'])

    # 随机打乱数据
    train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)

    # 分割数据
    split_ratio = 0.8
    train_size = int(len(train_data) * split_ratio)
    train_samples = train_data[:train_size]
    val_samples = train_data[train_size:]

    # 创建数据集和数据加载器
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = TextDataset(data_dir='data', samples=train_samples, tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                              collate_fn=text_only_collate_fn)

    val_dataset = TextDataset(data_dir='data', samples=val_samples, tokenizer=tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=text_only_collate_fn)

    # 模型、损失函数和优化器
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learn_rate, weight_decay=config.weight_decay)

    # 训练只使用文本数据的模型
    train_text_only_model(model, train_loader, val_loader, criterion, optimizer)
