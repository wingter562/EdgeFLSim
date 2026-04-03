import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import Tuple, List

from config import Config
from models.lenet import LeNet5
from core.device import Device
from core.server import Server
from core.scheduler import Scheduler
from utils.logger import Logger
from core.edge_server import EdgeServer

def create_heterogeneous_data(config: Config, devices: List[Device]) -> Tuple[List[Subset], DataLoader]:
    """创建非IID数据分布（按标签偏斜分配）"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    targets = train_dataset.targets.numpy()
    indices_by_class = {i: [] for i in range(10)}
    for idx, label in enumerate(targets):
        indices_by_class[label].append(idx)

    subsets = []
    samples_per_device = len(train_dataset) // config.num_devices

    for device_id in range(config.num_devices):
        device_indices = []
        remaining = samples_per_device

        # 决定该设备的数据分布
        if device_id < int(config.num_devices * config.data_iid_ratio):
            # IID：均匀分布
            class_distribution = np.ones(10) / 10
        else:
            # Non-IID：主要集中于少数类别
            main_classes = random.sample(range(10), random.randint(1, 3))
            class_distribution = np.zeros(10)
            for c in main_classes:
                class_distribution[c] = config.class_imbalance_factor
            class_distribution = class_distribution / class_distribution.sum()

        # 分配样本
        for class_idx in range(10):
            n_samples = int(samples_per_device * class_distribution[class_idx])
            if n_samples > 0 and indices_by_class[class_idx]:
                selected = random.sample(indices_by_class[class_idx],
                                         min(n_samples, len(indices_by_class[class_idx])))
                device_indices.extend(selected)
                for idx in selected:
                    indices_by_class[class_idx].remove(idx)
            remaining -= n_samples

        # 补全剩余样本（随机）
        if remaining > 0:
            all_remaining = []
            for indices in indices_by_class.values():
                all_remaining.extend(indices)
            if all_remaining:
                additional = random.sample(all_remaining, min(remaining, len(all_remaining)))
                device_indices.extend(additional)

        subsets.append(Subset(train_dataset, device_indices))

    return subsets, test_loader

def run(config: Config, save_dir: str = "results"):
    """主仿真流程（无回调）"""
    return run_with_callback(config, None, save_dir)

def run_with_callback(config, callback, save_dir="results"):
    # 设置随机种子
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.use_cuda:
        torch.cuda.manual_seed(config.seed)

    device = torch.device("cuda" if config.use_cuda else "cpu")
    print(f"Using device: {device}")

    # 创建设备（根据 GPU 比例）
    num_gpu = int(config.num_devices * config.gpu_ratio)
    num_cpu = config.num_devices - num_gpu
    device_types = ['gpu'] * num_gpu + ['cpu'] * num_cpu
    random.shuffle(device_types)
    devices = [Device(i, config, device_types[i]) for i in range(config.num_devices)]

    # 创建数据
    data_subsets, test_loader = create_heterogeneous_data(config, devices)

    # 初始化中心服务器、调度器、日志器
    server = Server(config)
    scheduler = Scheduler()
    logger = Logger(config, save_dir)

    # 初始评估
    init_acc, init_loss = server.evaluate(test_loader)
    print(f"Initial model accuracy: {init_acc:.2f}%")

    # 根据基站模式选择训练流程
    if config.base_station_mode == "multi":
        # ==================== 多基站模式：三级聚合 ====================
        num_edge_servers = getattr(config, 'num_edge_servers', 2)
        edge_servers = [EdgeServer(i, config) for i in range(num_edge_servers)]
        for i, dev in enumerate(devices):
            edge_servers[i % num_edge_servers].add_device(dev)

        for round_idx in range(config.num_rounds):
            # 重置累计值
            for es in edge_servers:
                es.reset_metrics()
            server.reset_metrics()

            # 1. 设备训练，上传到所属边缘服务器
            for es in edge_servers:
                for dev in es.devices:
                    data_loader = DataLoader(data_subsets[dev.id], batch_size=config.batch_size, shuffle=True)
                    state_dict, energy, time_cost, metrics = dev.train(
                        server.global_model, data_loader, device, round_idx
                    )
                    # 上传通信时间（上下行对称）
                    upload_time = config.model_size_bits / dev.bandwidth
                    upload_energy = dev.power * upload_time
                    es.receive_update(state_dict, len(data_subsets[dev.id]), dev.id,
                                      energy + upload_energy, time_cost + upload_time)

            # 2. 边缘服务器聚合，并发送到中心
            for es in edge_servers:
                es.aggregate_locally()
                # 边缘服务器到中心的通信（简化：使用固定带宽估算）
                upload_time_center = config.model_size_bits / (config.bandwidth_range[1] * 0.5)
                upload_energy_center = 1.0 * upload_time_center  # 假设边缘服务器功率 1W
                server.receive_edge_update(es.global_model.state_dict(), es.id,
                                           es.total_energy + upload_energy_center,
                                           es.total_time + upload_time_center)

            # 3. 中心服务器聚合
            server.aggregate_edge_updates()
            total_energy = server.total_energy
            total_time = server.total_time

            # 4. 全局评估
            acc, loss = server.evaluate(test_loader)
            # 记录日志（多基站模式下，selected_devices 可以传空列表）
            logger.update(round_idx+1, acc, loss, total_energy, total_time, selected_devices=[])

            print(f"Round {round_idx+1:2d}/{config.num_rounds:2d} | Acc: {acc:6.2f}% | Loss: {loss:.4f} | Energy: {total_energy:6.2f}J | Time: {total_time:6.2f}s")

            # 回调数据
            round_data = {
                'round': round_idx+1,
                'accuracy': acc,
                'loss': loss,
                'total_energy': total_energy,
                'total_time': total_time,
                'selected_devices': [],
                'accuracy_history': logger.history["accuracy"].copy(),
                'loss_history': logger.history["loss"].copy(),
                'energy_history': logger.history["total_energy"].copy(),
                'time_history': logger.history["total_time"].copy(),
                'devices': [{'id': d.id, 'type': d.device_type, 'freq': d.freq,
                             'bandwidth': d.bandwidth, 'k': d.k,
                             'participation_count': d.participation_count} for d in devices]
            }
            if callback:
                callback(round_data)

            # 检查点保存
            if config.save_checkpoints and (round_idx+1) % 10 == 0:
                checkpoint_path = f"{save_dir}/checkpoint_round_{round_idx+1}.pth"
                torch.save({
                    'round': round_idx+1,
                    'model_state_dict': server.global_model.state_dict(),
                    'accuracy': acc
                }, checkpoint_path)

    else:
        # ==================== 单基站模式：两级聚合 ====================
        for round_idx in range(config.num_rounds):
            # 选择设备
            selected = scheduler.select(devices, round_idx, config)
            client_updates = []
            sample_sizes = []
            client_losses = []
            total_energy = 0.0
            total_time = 0.0

            for dev in selected:
                data_loader = DataLoader(data_subsets[dev.id], batch_size=config.batch_size, shuffle=True)
                state_dict, energy, time_cost, metrics = dev.train(
                    server.global_model, data_loader, device, round_idx
                )
                client_updates.append(state_dict)
                sample_sizes.append(len(data_subsets[dev.id]))
                client_losses.append(metrics['local_losses'][-1])
                total_energy += energy
                total_time += time_cost

            # 聚合
            server.aggregate(client_updates, selected, sample_sizes, client_losses)

            # 全局评估
            acc, loss = server.evaluate(test_loader)
            logger.update(round_idx+1, acc, loss, total_energy, total_time, selected)

            print(f"Round {round_idx+1:2d}/{config.num_rounds:2d} | Acc: {acc:6.2f}% | Loss: {loss:.4f} | Energy: {total_energy:6.2f}J | Time: {total_time:6.2f}s")

            # 回调数据
            round_data = {
                'round': round_idx+1,
                'accuracy': acc,
                'loss': loss,
                'total_energy': total_energy,
                'total_time': total_time,
                'selected_devices': [d.id for d in selected],
                'accuracy_history': logger.history["accuracy"].copy(),
                'loss_history': logger.history["loss"].copy(),
                'energy_history': logger.history["total_energy"].copy(),
                'time_history': logger.history["total_time"].copy(),
                'devices': [{'id': d.id, 'type': d.device_type, 'freq': d.freq,
                             'bandwidth': d.bandwidth, 'k': d.k,
                             'participation_count': d.participation_count} for d in devices]
            }
            if callback:
                callback(round_data)

            # 检查点保存
            if config.save_checkpoints and (round_idx+1) % 10 == 0:
                checkpoint_path = f"{save_dir}/checkpoint_round_{round_idx+1}.pth"
                torch.save({
                    'round': round_idx+1,
                    'model_state_dict': server.global_model.state_dict(),
                    'accuracy': acc
                }, checkpoint_path)

    # 最终保存
    logger.save_data()
    report = logger.generate_report()
    with open(f"{save_dir}/final_report.txt", 'w') as f:
        f.write(report)
    print(report)

    import pandas as pd
    df = pd.DataFrame(logger.history)
    return df