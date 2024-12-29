import torch
from torch import nn
from torchvision import models
from torchvision.models.resnet import ResNet50_Weights
from transformers import BertModel


class MultiModalModel(nn.Module):
    def __init__(self, num_classes=3):
        super(MultiModalModel, self).__init__()
        self.image_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.image_model.fc = nn.Identity()

        # 解冻部分卷积层
        for param in self.image_model.parameters():
            param.requires_grad = True  # 解冻所有参数

        # 加载预训练的BERT模型，并解冻部分层
        self.text_model = BertModel.from_pretrained('bert-base-uncased')
        for param in self.text_model.parameters():
            param.requires_grad = True  # 解冻所有BERT层

        # 全连接层，输出情感类别
        self.fc = nn.Linear(2048 + 768, num_classes)
        self.dropout = nn.Dropout(p=0.1)  # Dropout层防止过拟合

    def forward(self, image, text_input):
        # 获取图像特征
        image_features = self.image_model(image)

        # 获取文本特征
        text_features = self.text_model(**text_input).pooler_output

        # 融合图像和文本特征
        combined_features = torch.cat((image_features, text_features), dim=1)

        # Dropout防止过拟合
        combined_features = self.dropout(combined_features)

        # 输出情感预测结果
        output = self.fc(combined_features)
        return output
