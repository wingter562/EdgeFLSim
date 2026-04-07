import torch.nn as nn
import torchvision.models as tv_models
from models.lenet import LeNet5
from models.simple_cnn import SimpleCNN


def get_model(model_name: str, input_channels=1, input_height=28, input_width=28, num_classes=10, **kwargs):
    model_name = model_name.lower()

    if model_name == 'lenet5':
        from models.lenet import LeNet5
        model = LeNet5()
        # 修改第一层卷积的输入通道
        model.conv1 = nn.Conv2d(input_channels, 6, kernel_size=5, padding=2)
        model.fc3 = nn.Linear(84, num_classes)
        return model

    elif model_name == 'simple_cnn':
        from models.simple_cnn import SimpleCNN
        # 如果你的 SimpleCNN 已支持动态尺寸，直接调用；否则使用下面修改版
        return SimpleCNN(input_channels, input_height, input_width, num_classes)

    elif model_name == 'resnet18':
        # 加载 ResNet18（不使用预训练权重）
        model = tv_models.resnet18(weights=None, num_classes=num_classes)
        # 修改第一层卷积：标准 ResNet18 第一层是 conv1 (in=3, out=64, kernel=7, stride=2, padding=3)
        # 对于 CIFAR (32x32)，我们改用 kernel=3, stride=1, padding=1，并移除后续的 maxpool
        model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # 将原来的 maxpool 层替换为 Identity（恒等映射），因为图像已经很小
        model.maxpool = nn.Identity()
        # 注意：ResNet18 的输入尺寸理论上可以是任意，但后续的卷积层 stride 均为 1 和 2，需要保证特征图尺寸不为负数。
        # 对于 32x32，经过 conv1 (stride=1) 输出 32x32，然后经过 layer1 (stride=1) 输出 32x32，
        # layer2 (stride=2) 输出 16x16，layer3 (stride=2) 输出 8x8，layer4 (stride=2) 输出 4x4，
        # 最后 adaptive_avg_pool 输出 1x1，完全可行。
        return model

    else:
        raise ValueError(f"Unsupported model: {model_name}")