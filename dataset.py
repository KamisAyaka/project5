# dataset.py
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import BertTokenizer

import config


class MultiModalDataset(Dataset):
    """
    多模态数据集类，继承自torch.utils.data.Dataset
    用于加载和处理包含文本和图像的多模态数据
    """

    def __init__(self, data_dir, samples, transform=None, is_test=False):
        """
        初始化多模态数据集
        :param data_dir: 数据目录，包含文本和图像数据
        :param samples: 数据样本，通常包含样本ID和标签
        :param transform: 可选的图像变换方法
        :param is_test: 是否为测试集，影响标签的加载
        """
        self.data_dir = data_dir
        self.samples = samples
        self.transform = transform
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # 创建标签到整数的映射
        self.label_to_int = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.int_to_label = {v: k for k, v in self.label_to_int.items()}
        self.is_test = is_test

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
        :return: 处理后的样本，包括图像、文本和标签（如果是训练集）
        """
        guid = self.samples.iloc[idx, 0]
        label = self.samples.iloc[idx, 1]

        # 如果是测试集，忽略标签
        if self.is_test:
            label = None
        else:
            # 将标签转换为整数
            label = self.label_to_int[label]

        # 读取文本文件
        text_path = os.path.join(self.data_dir, f"{guid}.txt")
        with open(text_path, 'r', encoding='utf-8', errors='replace') as file:
            text = file.read().strip()

        # 读取图像文件并进行初始化处理
        image_path = os.path.join(self.data_dir, f"{guid}.jpg")
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # 使用tokenizer对输入文本进行编码，返回包含输入序列的字典
        # return_tensors='pt'指示返回的是PyTorch张量，padding=True对序列进行填充以使它们长度相同
        # truncation=True在超过最大长度时截断序列，max_length设置最大序列长度
        text_input = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=config.context_max_len)

        # 移除字典中每个张量的冗余维度，确保每个张量的维度都是(1, -1)
        text_input = {key: val.squeeze(0) for key, val in text_input.items()}

        # 如果标签不为空，将标签转换为PyTorch长整型张量，并与图像和文本输入一起返回
        if label is not None:
            label = torch.tensor(label, dtype=torch.long)
            return image, text_input, label
        # 如果标签为空，仅返回图像和文本输入
        else:
            return image, text_input


