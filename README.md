# Lab5:多模态情感分析

#### 实验要求：

本项目是一个多模态情感分类任务，结合图像和文本数据进行情感分析。使用了预训练的AlexNet和BERT模型，并通过自定义的数据集和训练流程实现了模型的训练、验证和预测。

代码仓库在()

#### 文件结构：

```
project5/
├── alex_model.py        # 定义基于AlexNet和BERT的多模态模型
├── bert.py                  # 文本数据处理及仅基于文本的BERT模型训练
├── config.py                # 配置参数（如设备选择、超参数等）
├── dataset.py               # 多模态数据集类定义
├── predict.py               # 测试集预测脚本
├── Resnet50.py              # 基于ResNet50的图像分类模型训练
├── Resnet50_model.py        # 定义基于ResNet50和BERT的多模态模型
└── train.py                 # 多模态模型训练脚本
```

#### 执行代码的完整流程：
1. 环境准备

 确保安装了以下依赖库：
 ```
 pip install requirements.txt
 ```

2. 数据准备
- 将图像文件放置在data/目录下，文件名为{guid}.jpg。
- 将文本文件放置在data/目录下，文件名为{guid}.txt。
- 准备训练数据文件train.txt，格式为两列：guid, tag。
- 准备测试数据文件test_without_label.txt，格式为两列：guid, tag（标签可以为空）。

3. 训练多模态模型
使用 AlexNet 和 BERT 模型
   1. 运行 train.py 文件来训练多模态模型：
    ```
    python train.py
   ```
   - 该脚本会读取train.txt文件，随机打乱并分割成训练集和验证集。
        - 创建多模态数据集和数据加载器。
        - 初始化基于AlexNet和BERT的多模态模型。
        - 训练模型并在验证集上评估性能。
        - 保存最佳模型到best_model.pth
     
4. 测试集预测
    1. 运行 predict.py 文件来进行测试集预测：
    ```
       python predict.py
   ```
- 加载训练好的多模态模型（默认加载best_model.pth）。 
  - 读取test_without_label.txt文件中的测试数据。 
  - 创建多模态数据集和数据加载器。 
  - 使用模型进行预测并将结果保存回test_without_label.txt文件中。


5. 仅使用文本数据训练BERT模型
   1. 运行 bert.py 文件来训练仅基于文本的BERT模型：
       ```
         python bert.py
      ```
      - 该脚本会读取train.txt文件，随机打乱并分割成训练集和验证集。
        - 创建文本数据集和数据加载器。 
        - 初始化BERT模型进行序列分类。 
        - 训练模型并在验证集上评估性能。

6. 仅使用图片数据训练Resnet50模型
   1. 运行 Resnet50.py 文件来训练仅基于图片的Resnet50模型：
       ```
           python Resnet50.py
      ```
      - 该脚本会读取train.txt文件，随机打乱并分割成训练集和验证集。
        - 创建图片数据集和数据加载器。 
        - 初始化基于ResNet50的图像分类模型。
        - 训练模型并在验证集上评估性能。