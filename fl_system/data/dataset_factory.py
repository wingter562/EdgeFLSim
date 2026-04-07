# data/dataset_factory.py
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

def get_dataset(name: str, train: bool, data_dir: str = "./data"):
    """
    返回数据集对象及其元信息 (input_channels, input_height, input_width, num_classes)
    """
    name = name.lower()
    
    # 定义不同数据集的预处理
    if name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = torchvision.datasets.MNIST(data_dir, train=train, download=True, transform=transform)
        meta = {'channels': 1, 'height': 28, 'width': 28, 'classes': 10}
        
    elif name == 'fashion_mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = torchvision.datasets.FashionMNIST(data_dir, train=train, download=True, transform=transform)
        meta = {'channels': 1, 'height': 28, 'width': 28, 'classes': 10}
        
    elif name == 'emnist':
        # EMNIST 有多个版本，这里使用 'byclass' (62类，字母+数字)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = torchvision.datasets.EMNIST(data_dir, split='byclass', train=train, download=True, transform=transform)
        meta = {'channels': 1, 'height': 28, 'width': 28, 'classes': 62}
        
    elif name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        dataset = torchvision.datasets.CIFAR10(data_dir, train=train, download=True, transform=transform)
        meta = {'channels': 3, 'height': 32, 'width': 32, 'classes': 10}
        
    elif name == 'cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        dataset = torchvision.datasets.CIFAR100(data_dir, train=train, download=True, transform=transform)
        meta = {'channels': 3, 'height': 32, 'width': 32, 'classes': 100}
        
    elif name == 'svhn':
        # SVHN 是街景门牌号，10类（数字0-9）
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
        ])
        split = 'train' if train else 'test'
        dataset = torchvision.datasets.SVHN(data_dir, split=split, download=True, transform=transform)
        meta = {'channels': 3, 'height': 32, 'width': 32, 'classes': 10}
        
    else:
        raise ValueError(f"Unsupported dataset: {name}")
    
    return dataset, meta