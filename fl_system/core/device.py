import time
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, List, Any
from config import Config
from models.lenet import LeNet5
from models.model_factory import get_model
class Device:
    """边缘设备建模：包含硬件特性、动态波动、能耗计算、本地训练"""
    def __init__(self, device_id: int, config: Config, device_type: str = None):
        self.id = device_id
        self.config = config
        self.model_name = config.model_name
        self.input_channels = getattr(config, 'input_channels', 1)
        self.num_classes = getattr(config, 'num_classes', 10)
        self.input_height = getattr(config, 'input_height', 32)
        self.input_width = getattr(config, 'input_width', 32)
        if device_type is None:
            # 向后兼容：如果没有指定，随机选择
            self.device_type = random.choice(['cpu', 'gpu'])
        else:
            self.device_type = device_type

        self._init_characteristics()  # 根据 self.device_type 初始化频率、k 等

        # 状态
        self.last_energy = 0.0
        self.last_accuracy = 0.0
        self.participation_count = 0
        self.total_energy = 0.0
        self.total_time = 0.0

        # 历史
        self.energy_history = []
        self.accuracy_history = []

        #位置
        self.x = random.uniform(0, 100)
        self.y = random.uniform(0, 100)
        self.speed = random.uniform(0, 5)  # 每轮移动距离
        self.angle = random.uniform(0, 2 * math.pi)


    def _init_characteristics(self):
        """初始化硬件特性（静态基准值）"""
        if self.device_type == 'cpu':
            self.base_freq = random.uniform(*self.config.cpu_freq_range)
            self.base_compute_capacity = random.uniform(*self.config.compute_capacity_range)
            self.k = random.uniform(*self.config.k_cpu_range)  # CPU取上限范围
        else:
            self.base_freq = random.uniform(*self.config.gpu_freq_range)
            self.base_compute_capacity = random.uniform(*self.config.compute_capacity_range)
            self.k = random.uniform(*self.config.k_gpu_range)  # GPU取下限范围
        self.base_bandwidth = random.uniform(*self.config.bandwidth_range)
        self.power = random.uniform(*self.config.power_range)

        # 当前值（初始等于基准值）
        self.freq = self.base_freq
        self.bandwidth = self.base_bandwidth
        self.compute_capacity = self.base_compute_capacity

    def update_dynamic_characteristics(self):
        """每轮动态波动：频率、带宽、计算能力在基准值附近随机变化"""
        fluctuation = self.config.dynamic_fluctuation
        self.freq = self.base_freq * random.uniform(1 - fluctuation, 1 + fluctuation)
        self.bandwidth = self.base_bandwidth * random.uniform(1 - fluctuation, 1 + fluctuation)
        self.compute_capacity = self.base_compute_capacity * random.uniform(1 - fluctuation, 1 + fluctuation)

    def compute_score(self) -> float:
        """计算选择分数（能量效率+准确率）"""
        if self.participation_count == 0:
            return float('inf')
        energy_efficiency = 1.0 / (self.last_energy + 1e-6)
        accuracy_score = self.last_accuracy / 100.0
        return (self.config.energy_weight * energy_efficiency +
                self.config.accuracy_weight * accuracy_score)

    def compute_energy(self, compute_time: float, upload_time: float, download_time: float) -> float:
        e_comp = self.k * (self.freq ** 2) * compute_time
        e_upload = self.power * upload_time
        e_download = self.power * download_time
        return e_comp + e_upload + e_download

    def train(self, global_model: nn.Module, data_loader: torch.utils.data.DataLoader,
              device: torch.device, round_idx: int) -> Tuple[Dict, float, float, Dict]:
        """本地训练，返回本地模型参数、能耗、时间、指标"""
        # 动态更新硬件特性
        self.update_dynamic_characteristics()

        # 复制全局模型
        local_model = get_model(
            self.model_name,
            input_channels=self.input_channels,
            input_height=self.input_height,
            input_width=self.input_width,
            num_classes=self.num_classes
        ).to(device)
        local_model.load_state_dict(global_model.state_dict())

        optimizer = optim.SGD(local_model.parameters(), lr=self.config.learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        # FedProx 正则化参数
        mu = 0.01 if self.config.aggregation_method == "fedprox" else 0.0
        global_params = {k: v.detach().clone() for k, v in global_model.named_parameters()} if mu > 0 else None

        # 计时
        start_time = time.time()

        local_model.train()
        local_losses = []
        local_accuracies = []

        for epoch in range(self.config.local_epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            for batch_idx, (x, y) in enumerate(data_loader):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                output = local_model(x)
                loss = loss_fn(output, y)

                if mu > 0:
                    prox_reg = 0.0
                    for name, param in local_model.named_parameters():
                        prox_reg += torch.norm(param - global_params[name]) ** 2
                    loss += (mu / 2) * prox_reg

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

            avg_loss = epoch_loss / len(data_loader)
            avg_acc = 100.0 * correct / total
            local_losses.append(avg_loss)
            local_accuracies.append(avg_acc)

        # 计算时间
        compute_time = time.time() - start_time
        # 通信时间：模型上传 + 下载（考虑动态带宽）
        upload_time = self.config.model_size_bits / self.bandwidth
        download_time = self.config.model_size_bits / self.bandwidth   # 假设上下行对称
        total_time = compute_time + upload_time + download_time

        # 能耗（分解计算）
        e_comp = self.k * (self.freq ** 2) * compute_time
        e_upload = self.power * upload_time
        e_download = self.power * download_time
        total_energy = e_comp + e_upload + e_download

        # 更新状态
        self.last_energy = total_energy
        self.last_accuracy = local_accuracies[-1]
        self.participation_count += 1
        self.total_energy += total_energy
        self.total_time += total_time
        self.energy_history.append(total_energy)
        self.accuracy_history.append(self.last_accuracy)

        metrics = {
            "local_losses": local_losses,
            "local_accuracies": local_accuracies,
            "compute_time": compute_time,
            "upload_time": upload_time,
            "download_time": download_time,
            "total_time": total_time,
            "energy": total_energy,
            "comm_energy": e_upload + e_download,  # 通信能耗
            "comp_energy": e_comp,  # 计算能耗
        }
        return local_model.state_dict(), total_energy, total_time, metrics