import random
import numpy as np
from typing import List
from core.device import Device
from config import Config

class Scheduler:
    """设备选择策略"""
    @staticmethod
    def select(devices: List[Device], round_idx: int, config: Config) -> List[Device]:
        num_to_select = int(config.num_devices * config.selection_ratio)
        strategy = config.selection_strategy

        if strategy == "random":
            return random.sample(devices, num_to_select)
        elif strategy == "energy_aware":
            sorted_devices = sorted(devices, key=lambda d: d.compute_score(), reverse=True)
            return sorted_devices[:num_to_select]
        elif strategy == "capability_aware":
            sorted_devices = sorted(devices, key=lambda d: d.compute_capacity, reverse=True)
            return sorted_devices[:num_to_select]
        elif strategy == "hybrid":
            # 计算所有设备的最大能耗，用于归一化
            max_energy = max(d.last_energy for d in devices) if devices else 1
            scores = []
            for d in devices:
                # 能量分数：能耗越低，分数越高（范围 0~1）
                energy_score = 1.0 - (d.last_energy / (max_energy + 1e-6))
                capability_score = d.compute_capacity  # 范围 0.5~1.5
                fairness_score = 1.0 / (d.participation_count + 1)  # 范围 0~1
                total_score = (0.4 * energy_score + 0.3 * capability_score + 0.3 * fairness_score)
                scores.append(total_score)
            indices = np.argsort(scores)[::-1]
            return [devices[i] for i in indices[:num_to_select]]
        else:
            raise ValueError(f"Unknown selection strategy: {strategy}")