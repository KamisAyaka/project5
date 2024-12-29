import torch
from torch import nn
from torchvision import models
from transformers import BertModel
from torchvision.models import AlexNet_Weights


class MultiModalModel(nn.Module):
    def __init__(self, num_classes=3):
        super(MultiModalModel, self).__init__()

        # 使用预训练的AlexNet模型
        self.image_model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
        self.image_model.classifier[6] = nn.Identity()  # 移除最后的全连接层

        # 使用BERT模型进行文本处理
        self.text_model = BertModel.from_pretrained('bert-base-uncased')

        # 全连接层，用于情感分类
        # 4096 (AlexNet输出的特征维度) + 768 (BERT pooler_output维度)
        self.fc = nn.Linear(4096 + 768, num_classes)
        self.dropout = nn.Dropout(p=0.1)  # Dropout防止过拟合

    def forward(self, image, text_input):
        # 获取图像特征
        image_features = self.image_model(image)

        # 获取文本特征
        text_features = self.text_model(**text_input).pooler_output

        # 融合图像和文本特征
        combined_features = torch.cat((image_features, text_features), dim=1)

        # Dropout防止过拟合
        combined_features = self.dropout(combined_features)

        # 最终分类输出
        output = self.fc(combined_features)
        return output
