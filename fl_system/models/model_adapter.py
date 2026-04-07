import torch.nn as nn

def adapt_model(model, input_channels, num_classes):
    """
    动态调整模型的第一层卷积（Conv2d）和最后一层全连接（Linear）
    以适配不同的输入通道数和输出类别数。
    适用于常见的 CNN 架构，如 LeNet5、SimpleCNN、ResNet 等。
    """
    # 1. 找到第一个 Conv2d 层并修改其 in_channels
    first_conv = None
    first_conv_name = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            first_conv = module
            first_conv_name = name
            break

    if first_conv is not None and first_conv.in_channels != input_channels:
        # 创建新的卷积层，保持其他参数不变
        new_conv = nn.Conv2d(
            input_channels,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None
        )
        # 初始化新卷积层（可选）
        nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
        if new_conv.bias is not None:
            nn.init.constant_(new_conv.bias, 0)
        # 替换原层
        _replace_submodule(model, first_conv_name, new_conv)

    # 2. 找到最后一个 Linear 层并修改其 out_features
    last_linear = None
    last_linear_name = None
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Linear):
            last_linear = module
            last_linear_name = name
            break

    if last_linear is not None and last_linear.out_features != num_classes:
        new_linear = nn.Linear(last_linear.in_features, num_classes)
        nn.init.xavier_uniform_(new_linear.weight)
        if new_linear.bias is not None:
            nn.init.constant_(new_linear.bias, 0)
        _replace_submodule(model, last_linear_name, new_linear)

    return model

def _replace_submodule(module, submodule_path, new_submodule):
    """辅助函数：根据点分隔的路径替换模块"""
    parts = submodule_path.split('.')
    parent = module
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_submodule)