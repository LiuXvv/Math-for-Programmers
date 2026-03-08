# Math for Programmers

本仓库收集程序员学习数学的相关资料和代码实现。

---

## AlexNet (深度学习 CNN)

> ImageNet Classification with Deep Convolutional Neural Networks (2012)

### 论文信息

- **作者**: Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton
- **会议**: NeurIPS 2012 (原 NIPS 2012)
- **论文链接**: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-neural-networks

### 核心贡献

1. **首次**在大规模图像分类中使用 ReLU 激活函数
2. 使用 Dropout 防止过拟合
3. 使用 Local Response Normalization (LRN)
4. 数据增强 (随机裁剪、水平翻转)
5. GPU 训练 (当时用 2 张 GTX 580)

### 网络结构

```
输入 (224x224x3)
    ↓
Conv1 (11x11, 96 filters, stride=4) → ReLU → LRN → MaxPool
    ↓
Conv2 (5x5, 256 filters) → ReLU → LRN → MaxPool
    ↓
Conv3 (3x3, 384 filters) → ReLU
    ↓
Conv4 (3x3, 384 filters) → ReLU
    ↓
Conv5 (3x3, 256 filters) → ReLU → MaxPool
    ↓
AdaptiveAvgPool (6x6)
    ↓
Flatten
    ↓
FC1 (4096) → ReLU → Dropout
    ↓
FC2 (4096) → ReLU → Dropout
    ↓
FC3 (1000) → Output
```

### 参数统计

- **总参数量**: 约 61M (6100 万)
- **卷积层**: 约 3.7M
- **全连接层**: 约 57M

### 使用方法

```python
import torch
from alexnet import alexnet

# 创建模型
model = alexnet(num_classes=1000)

# 测试输入
x = torch.randn(1, 3, 224, 224)
output = model(x)
print(output.shape)  # torch.Size([1, 1000])

# 迁移学习 (修改输出类别)
model = alexnet(num_classes=10)  # 10 类分类
```

### 运行测试

```bash
python alexnet.py
```

### 依赖

```
torch>=1.0
```

### 参考

- [原版论文](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
- [ImageNet](https://www.image-net.org/)
- [PyTorch AlexNet](https://pytorch.org/hub/pytorch_vision_alexnet/)