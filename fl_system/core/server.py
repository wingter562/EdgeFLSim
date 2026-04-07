import torch
import torch.nn as nn
from typing import List, Dict, Tuple
from models.lenet import LeNet5
from core.device import Device
from models.model_factory import get_model

class Server:
    """中心服务器：模型聚合、评估"""
    def __init__(self, config):
        self.config = config
        input_channels = getattr(config, 'input_channels', 3)
        input_height = getattr(config, 'input_height', 32)
        input_width = getattr(config, 'input_width', 32)
        num_classes = getattr(config, 'num_classes', 10)

        self.global_model = get_model(
            config.model_name,
            input_channels=input_channels,
            input_height=input_height,
            input_width=input_width,
            num_classes=num_classes
        )
        if config.use_cuda:
            self.global_model = self.global_model.cuda()
        self.device = torch.device("cuda" if config.use_cuda else "cpu")
        self.edge_updates = []
        self.total_energy = 0.0
        self.total_time = 0.0

    def receive_edge_update(self, state_dict, edge_id, edge_energy, edge_time):
        self.total_energy += edge_energy
        self.total_time += edge_time
        self.edge_updates.append((state_dict, edge_id))

    def aggregate_edge_updates(self):
        """聚合所有边缘服务器的更新，生成全局模型"""
        if not self.edge_updates:
            return
        total_weight = len(self.edge_updates)
        global_dict = self.global_model.state_dict()
        for key in global_dict:
            # 只对浮点型参数进行加权平均
            if global_dict[key].dtype in (torch.float32, torch.float64, torch.float16):
                global_dict[key] = torch.zeros_like(global_dict[key], dtype=torch.float32)
                for update, _ in self.edge_updates:
                    global_dict[key] += update[key].float() / total_weight
            else:
                # 非浮点型参数（如整型计数器）直接取第一个边缘服务器的值
                if self.edge_updates:
                    global_dict[key] = self.edge_updates[0][0][key].clone()
        self.global_model.load_state_dict(global_dict)
        self.edge_updates.clear()
        self.total_energy += 0.2
        self.total_time += 0.02

    def reset_metrics(self):
        self.total_energy = 0.0
        self.total_time = 0.0
        self.edge_updates.clear()

    def aggregate(self, client_updates, devices, sample_sizes, client_losses):
        method = self.config.aggregation_method
        if method == "fedavg":
            self._fedavg(client_updates, sample_sizes)
        elif method == "fedavg_energy":
            self._fedavg_energy(client_updates, devices)
        elif method == "fedprox":
            self._fedavg(client_updates, sample_sizes)
        elif method == "qfed":
            self._qfed(client_updates, client_losses)
        elif method == "capability_weighted":
            self._capability_weighted(client_updates, devices, sample_sizes)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def _capability_weighted(self, client_updates, devices, sample_sizes):
        weights = [d.compute_capacity * sample_sizes[i] for i, d in enumerate(devices)]
        total_weight = sum(weights)
        global_dict = self.global_model.state_dict()
        for key in global_dict:
            if global_dict[key].dtype in (torch.float32, torch.float64, torch.float16):
                global_dict[key] = torch.zeros_like(global_dict[key], dtype=torch.float32)
                for i, update in enumerate(client_updates):
                    weight = weights[i] / total_weight
                    update_val = update[key].float() if update[key].is_floating_point() else update[key].float()
                    global_dict[key] += weight * update_val
            else:
                if client_updates:
                    global_dict[key] = client_updates[0][key].clone()
        self.global_model.load_state_dict(global_dict)

    def _fedavg(self, client_updates: List[Dict], sample_sizes: List[int]) -> None:
        total_samples = sum(sample_sizes)
        global_dict = self.global_model.state_dict()
        for key in global_dict:
            if global_dict[key].dtype in (torch.float32, torch.float64, torch.float16):
                global_dict[key] = torch.zeros_like(global_dict[key], dtype=torch.float32)
                for i, update in enumerate(client_updates):
                    weight = sample_sizes[i] / total_samples
                    update_val = update[key].float() if update[key].is_floating_point() else update[key].float()
                    global_dict[key] += weight * update_val
            else:
                if client_updates:
                    global_dict[key] = client_updates[0][key].clone()
        self.global_model.load_state_dict(global_dict)

    def _fedavg_energy(self, client_updates: List[Dict], devices: List[Device]) -> None:
        weights = [1.0 / (d.last_energy + 1e-6) for d in devices]
        total_weight = sum(weights)
        global_dict = self.global_model.state_dict()
        for key in global_dict:
            if global_dict[key].dtype in (torch.float32, torch.float64, torch.float16):
                global_dict[key] = torch.zeros_like(global_dict[key], dtype=torch.float32)
                for i, update in enumerate(client_updates):
                    weight = weights[i] / total_weight
                    update_val = update[key].float() if update[key].is_floating_point() else update[key].float()
                    global_dict[key] += weight * update_val
            else:
                if client_updates:
                    global_dict[key] = client_updates[0][key].clone()
        self.global_model.load_state_dict(global_dict)

    def _qfed(self, client_updates: List[Dict], client_losses: List[float], q: float = 1.0) -> None:
        weights = [loss ** q for loss in client_losses]
        total_weight = sum(weights)
        global_dict = self.global_model.state_dict()
        for key in global_dict:
            if global_dict[key].dtype in (torch.float32, torch.float64, torch.float16):
                global_dict[key] = torch.zeros_like(global_dict[key], dtype=torch.float32)
                for i, update in enumerate(client_updates):
                    weight = weights[i] / total_weight
                    update_val = update[key].float() if update[key].is_floating_point() else update[key].float()
                    global_dict[key] += weight * update_val
            else:
                if client_updates:
                    global_dict[key] = client_updates[0][key].clone()
        self.global_model.load_state_dict(global_dict)

    def evaluate(self, test_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        self.global_model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        loss_fn = nn.CrossEntropyLoss()
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.global_model(x)
                loss = loss_fn(outputs, y)
                total_loss += loss.item() * x.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        accuracy = 100.0 * correct / total
        avg_loss = total_loss / total
        return accuracy, avg_loss