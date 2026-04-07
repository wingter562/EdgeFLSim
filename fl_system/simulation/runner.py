import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import Tuple, List
import time

from config import Config
from models.model_factory import get_model
from data.dataset_factory import get_dataset
from models.lenet import LeNet5
from core.device import Device
from core.server import Server
from core.scheduler import Scheduler
from utils.logger import Logger
from core.edge_server import EdgeServer


def create_heterogeneous_data(config: Config) -> Tuple[List[Subset], DataLoader]:
    """创建非IID数据分布（按标签偏斜分配），支持插拔式数据集"""
    train_dataset, meta = get_dataset(config.dataset_name, train=True, data_dir="./data")
    test_dataset, _ = get_dataset(config.dataset_name, train=False, data_dir="./data")
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    num_classes = meta['classes']
    config.num_classes = num_classes
    config.input_channels = meta['channels']
    config.input_height = meta['height']
    config.input_width = meta['width']

    if hasattr(train_dataset, 'targets'):
        targets = np.array(train_dataset.targets)
    elif hasattr(train_dataset, 'labels'):
        targets = np.array(train_dataset.labels)
    else:
        raise AttributeError("Dataset has no 'targets' or 'labels' attribute")

    indices_by_class = {i: [] for i in range(num_classes)}
    for idx, label in enumerate(targets):
        indices_by_class[label].append(idx)

    subsets = []
    samples_per_device = len(train_dataset) // config.num_devices

    for device_id in range(config.num_devices):
        device_indices = []
        remaining = samples_per_device

        if device_id < int(config.num_devices * config.data_iid_ratio):
            class_distribution = np.ones(num_classes) / num_classes
        else:
            num_main = random.randint(1, min(3, num_classes))
            main_classes = random.sample(range(num_classes), num_main)
            class_distribution = np.zeros(num_classes)
            for c in main_classes:
                class_distribution[c] = config.class_imbalance_factor
            class_distribution = class_distribution / class_distribution.sum()

        for class_idx in range(num_classes):
            n_samples = int(samples_per_device * class_distribution[class_idx])
            if n_samples > 0 and indices_by_class[class_idx]:
                selected = random.sample(indices_by_class[class_idx],
                                         min(n_samples, len(indices_by_class[class_idx])))
                device_indices.extend(selected)
                for idx in selected:
                    indices_by_class[class_idx].remove(idx)
            remaining -= n_samples

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
    return run_with_callback(config, None, save_dir)


def run_with_callback(config, callback, save_dir="results"):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.use_cuda:
        torch.cuda.manual_seed(config.seed)

    device = torch.device("cuda" if config.use_cuda else "cpu")
    print(f"Using device: {device}")

    # 1. 加载数据集
    t0 = time.time()
    data_subsets, test_loader = create_heterogeneous_data(config)
    t1 = time.time()
    #print(f"[Timing] Data loading & partitioning took {t1-t0:.2f} seconds")

    # 2. 创建设备
    t0 = time.time()
    num_gpu = int(config.num_devices * config.gpu_ratio)
    num_cpu = config.num_devices - num_gpu
    device_types = ['gpu'] * num_gpu + ['cpu'] * num_cpu
    random.shuffle(device_types)
    devices = [Device(i, config, device_types[i]) for i in range(config.num_devices)]
    t1 = time.time()
    #print(f"[Timing] Creating {len(devices)} devices took {t1-t0:.2f} seconds")

    # 3. 初始化服务器、调度器、日志器
    t0 = time.time()
    server = Server(config)
    scheduler = Scheduler()
    logger = Logger(config, save_dir)
    t1 = time.time()
    #print(f"[Timing] Server/scheduler/logger init took {t1-t0:.2f} seconds")

    # 4. 初始评估
    t0 = time.time()
    init_acc, init_loss = server.evaluate(test_loader)
    t1 = time.time()
    #print(f"[Timing] Initial evaluation took {t1-t0:.2f} seconds")
    print(f"Initial model accuracy: {init_acc:.2f}%")

    # 根据基站模式选择训练流程
    if config.base_station_mode == "multi":
        # ---------- 多基站模式（三级聚合） ----------
        num_edge_servers = getattr(config, 'num_edge_servers', 2)
        edge_servers = [EdgeServer(i, config) for i in range(num_edge_servers)]
        for i, dev in enumerate(devices):
            edge_servers[i % num_edge_servers].add_device(dev)

        for round_idx in range(config.num_rounds):
            round_start = time.time()

            for es in edge_servers:
                es.reset_metrics()
            server.reset_metrics()

            # 1. 设备训练，上传到边缘服务器
            train_start = time.time()
            device_details = []  # 收集本轮设备明细
            for es in edge_servers:
                for dev in es.devices:
                    data_loader = DataLoader(data_subsets[dev.id], batch_size=config.batch_size, shuffle=True)
                    state_dict, energy, time_cost, metrics = dev.train(
                        server.global_model, data_loader, device, round_idx
                    )
                    upload_time = config.model_size_bits / dev.bandwidth
                    upload_energy = dev.power * upload_time
                    device_total_energy = energy + upload_energy
                    device_total_time = time_cost + upload_time

                    # 记录设备级明细
                    logger.add_device_metrics(round_idx + 1, dev.id, metrics)
                    # 收集用于前端展示的明细
                    device_details.append({
                        'round': round_idx + 1,
                        'device_id': dev.id,
                        'compute_time': metrics['compute_time'],
                        'comm_time': metrics['upload_time'] + metrics['download_time'],
                        'comp_energy': metrics['comp_energy'],
                        'comm_energy': metrics['comm_energy']
                    })

                    es.receive_update(state_dict, len(data_subsets[dev.id]), dev.id,
                                      device_total_energy, device_total_time)
            train_end = time.time()
            #print(f"[Timing] Round {round_idx+1} device training took {train_end - train_start:.2f}s")

            # 2. 边缘服务器聚合，并发送到中心
            agg_start = time.time()
            for es in edge_servers:
                es.aggregate_locally()
                upload_time_center = config.model_size_bits / (config.bandwidth_range[1] * 0.5)
                upload_energy_center = 1.0 * upload_time_center
                server.receive_edge_update(es.global_model.state_dict(), es.id,
                                           es.total_energy + upload_energy_center,
                                           es.total_time + upload_time_center)
            server.aggregate_edge_updates()
            agg_end = time.time()
            #print(f"[Timing] Round {round_idx+1} aggregation took {agg_end - agg_start:.2f}s")

            total_energy = server.total_energy
            total_time = server.total_time

            # 全局评估
            eval_start = time.time()
            acc, loss = server.evaluate(test_loader)
            eval_end = time.time()
            #print(f"[Timing] Round {round_idx+1} evaluation took {eval_end - eval_start:.2f}s")

            logger.update(round_idx + 1, acc, loss, total_energy, total_time, selected_devices=[])

            # 网络流量统计
            device_traffic = 2 * config.model_size_bits * len(devices)
            edge_traffic = 2 * config.model_size_bits * len(edge_servers)
            total_traffic = device_traffic + edge_traffic
            logger.add_traffic(total_traffic)

            round_end = time.time()
            print(f"Round {round_idx+1:2d}/{config.num_rounds:2d} | Acc: {acc:6.2f}% | Loss: {loss:.4f} | "
                  f"Energy: {total_energy:6.2f}J | Time: {total_time:6.2f}s | Traffic: {total_traffic/1e6:.2f}Mb | "
                  f"Wall time: {round_end - round_start:.2f}s")

            # 回调数据（增加流量历史、公平性历史、设备明细）
            round_data = {
                'round': round_idx + 1,
                'accuracy': acc,
                'loss': loss,
                'total_energy': total_energy,
                'total_time': total_time,
                'selected_devices': [],
                'accuracy_history': logger.history["accuracy"].copy(),
                'loss_history': logger.history["loss"].copy(),
                'energy_history': logger.history["total_energy"].copy(),
                'time_history': logger.history["total_time"].copy(),
                'traffic_history': logger.traffic_history.copy(),
                'fairness_history': logger.history["fairness"].copy(),
                'device_details': device_details,
                'devices': [{'id': d.id, 'type': d.device_type, 'freq': d.freq,
                             'bandwidth': d.bandwidth, 'k': d.k,
                             'participation_count': d.participation_count} for d in devices]
            }
            if callback:
                callback(round_data)

            if config.save_checkpoints and (round_idx + 1) % 10 == 0:
                checkpoint_path = f"{save_dir}/checkpoint_round_{round_idx+1}.pth"
                torch.save({
                    'round': round_idx + 1,
                    'model_state_dict': server.global_model.state_dict(),
                    'accuracy': acc
                }, checkpoint_path)

    else:
        # ---------- 单基站模式（两级聚合） ----------
        for round_idx in range(config.num_rounds):
            round_start = time.time()

            selected = scheduler.select(devices, round_idx, config)
            client_updates = []
            sample_sizes = []
            client_losses = []
            total_energy = 0.0
            total_time = 0.0
            device_details = []  # 收集本轮设备明细

            train_start = time.time()
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
                logger.add_device_metrics(round_idx + 1, dev.id, metrics)
                # 收集明细
                device_details.append({
                    'round': round_idx + 1,
                    'device_id': dev.id,
                    'compute_time': metrics['compute_time'],
                    'comm_time': metrics['upload_time'] + metrics['download_time'],
                    'comp_energy': metrics['comp_energy'],
                    'comm_energy': metrics['comm_energy']
                })
            train_end = time.time()
            #print(f"[Timing] Round {round_idx+1} device training ({len(selected)} devices) took {train_end - train_start:.2f}s")

            agg_start = time.time()
            server.aggregate(client_updates, selected, sample_sizes, client_losses)
            agg_end = time.time()
            #print(f"[Timing] Round {round_idx+1} aggregation took {agg_end - agg_start:.2f}s")

            eval_start = time.time()
            acc, loss = server.evaluate(test_loader)
            eval_end = time.time()
            #print(f"[Timing] Round {round_idx+1} evaluation took {eval_end - eval_start:.2f}s")

            logger.update(round_idx + 1, acc, loss, total_energy, total_time, selected)

            total_traffic = 2 * config.model_size_bits * len(selected)
            logger.add_traffic(total_traffic)

            round_end = time.time()
            print(f"Round {round_idx+1:2d}/{config.num_rounds:2d} | Acc: {acc:6.2f}% | Loss: {loss:.4f} | "
                  f"Energy: {total_energy:6.2f}J | Time: {total_time:6.2f}s | Traffic: {total_traffic/1e6:.2f}Mb | "
                  f"Wall time: {round_end - round_start:.2f}s")

            round_data = {
                'round': round_idx + 1,
                'accuracy': acc,
                'loss': loss,
                'total_energy': total_energy,
                'total_time': total_time,
                'selected_devices': [d.id for d in selected],
                'accuracy_history': logger.history["accuracy"].copy(),
                'loss_history': logger.history["loss"].copy(),
                'energy_history': logger.history["total_energy"].copy(),
                'time_history': logger.history["total_time"].copy(),
                'traffic_history': logger.traffic_history.copy(),
                'fairness_history': logger.history["fairness"].copy(),
                'device_details': device_details,
                'devices': [{'id': d.id, 'type': d.device_type, 'freq': d.freq,
                             'bandwidth': d.bandwidth, 'k': d.k,
                             'participation_count': d.participation_count} for d in devices]
            }
            if callback:
                callback(round_data)

            if config.save_checkpoints and (round_idx + 1) % 10 == 0:
                checkpoint_path = f"{save_dir}/checkpoint_round_{round_idx+1}.pth"
                torch.save({
                    'round': round_idx + 1,
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