# 检查是否有可用的 GPU
import torch
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs = 3
batch_size = 16
learn_rate = 2e-5
weight_decay = 0.01

context_max_len = 128

# 数据预处理
# 使用transforms.Compose组合多个变换操作，目的是将图像数据转换为适合模型输入的格式
transform = transforms.Compose([
    # 将图像大小调整为224x224像素，统一图像输入尺寸
    transforms.Resize((224, 224)),
    # 将图像转换为Tensor格式，以便于在PyTorch模型中使用
    transforms.ToTensor(),
    # 对图像进行标准化处理，减去均值并除以标准差，以加速模型训练和提高性能
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

