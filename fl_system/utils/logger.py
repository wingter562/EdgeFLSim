import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
from config import Config
from core.device import Device

sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class Logger:
    def __init__(self, config: Config, save_dir: str = "results"):
        self.config = config
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # 全局历史
        self.history = {
            "round": [],
            "accuracy": [],
            "loss": [],
            "total_energy": [],
            "total_time": [],
            "selected_devices": [],
            "fairness": []
        }

        # 设备级明细（新增）
        self.device_compute_time = {}   # device_id -> list of compute_time per round
        self.device_comm_time = {}      # device_id -> list of communication time (upload+download)
        self.device_comp_energy = {}    # device_id -> list of computation energy
        self.device_comm_energy = {}    # device_id -> list of communication energy
        self.device_rounds = {}         # device_id -> list of rounds participated

        # 网络流量历史（新增）
        self.traffic_history = []       # list of total traffic (bits) per round

        # 原有设备能耗/准确率记录（保持不变）
        self.device_energies = {}
        self.device_accuracies = {}

    def update(self, round_idx: int, accuracy: float, loss: float,
               total_energy: float, total_time: float,
               selected_devices: List[Device]) -> None:
        """更新全局历史"""
        self.history["round"].append(round_idx)
        self.history["accuracy"].append(accuracy)
        self.history["loss"].append(loss)
        self.history["total_energy"].append(total_energy)
        self.history["total_time"].append(total_time)
        self.history["selected_devices"].append([d.id for d in selected_devices])

        # 原有设备能耗/准确率记录
        for d in selected_devices:
            if d.id not in self.device_energies:
                self.device_energies[d.id] = []
                self.device_accuracies[d.id] = []
            self.device_energies[d.id].append(d.last_energy)
            self.device_accuracies[d.id].append(d.last_accuracy)

        # 公平性计算
        # 使用所有设备的参与次数（全局公平性）
        # 在 update 方法中，计算公平性
        all_counts = [d.participation_count for d in selected_devices]
        if len(all_counts) > 1:
            fairness = np.std(all_counts)
        else:
            fairness = 0.0
        self.history["fairness"].append(fairness)

    def add_device_metrics(self, round_idx: int, device_id: int, metrics: Dict):
        """记录单个设备的详细指标（计算时间、通信时间、能耗分解）"""
        if device_id not in self.device_compute_time:
            self.device_compute_time[device_id] = []
            self.device_comm_time[device_id] = []
            self.device_comp_energy[device_id] = []
            self.device_comm_energy[device_id] = []
            self.device_rounds[device_id] = []

        self.device_compute_time[device_id].append(metrics['compute_time'])
        self.device_comm_time[device_id].append(metrics['upload_time'] + metrics['download_time'])
        self.device_comp_energy[device_id].append(metrics['comp_energy'])
        self.device_comm_energy[device_id].append(metrics['comm_energy'])
        self.device_rounds[device_id].append(round_idx)

    def add_traffic(self, total_bits: float):
        """记录每轮网络总流量（比特）"""
        self.traffic_history.append(total_bits)

    def save_data(self):
        """保存所有历史数据到 CSV 文件"""
        # 1. 全局训练结果
        df = pd.DataFrame({
            'round': self.history["round"],
            'accuracy': self.history["accuracy"],
            'loss': self.history["loss"],
            'total_energy': self.history["total_energy"],
            'total_time': self.history["total_time"],
            'fairness': self.history["fairness"]
        })
        df.to_csv(f"{self.save_dir}/training_results.csv", index=False)

        # 2. 设备能耗明细（原有）
        n_rounds = len(self.history["round"])
        device_energy_df = pd.DataFrame(index=range(1, n_rounds+1), columns=list(self.device_energies.keys()))
        device_energy_df = device_energy_df.fillna(np.nan)
        for d_id, rounds in self.device_rounds.items():
            energies = self.device_energies.get(d_id, [])
            for r, e in zip(rounds, energies):
                device_energy_df.loc[r, d_id] = e
        device_energy_df.to_csv(f"{self.save_dir}/device_energies.csv")

        # 3. 设备级详细指标（新增）
        device_detail_rows = []
        for d_id in self.device_compute_time.keys():
            rounds = self.device_rounds[d_id]
            compute_times = self.device_compute_time[d_id]
            comm_times = self.device_comm_time[d_id]
            comp_energies = self.device_comp_energy[d_id]
            comm_energies = self.device_comm_energy[d_id]
            for r, ct, cmt, ce, cme in zip(rounds, compute_times, comm_times, comp_energies, comm_energies):
                device_detail_rows.append({
                    'device_id': d_id,
                    'round': r,
                    'compute_time_s': ct,
                    'comm_time_s': cmt,
                    'comp_energy_J': ce,
                    'comm_energy_J': cme
                })
        if device_detail_rows:
            device_detail_df = pd.DataFrame(device_detail_rows)
            device_detail_df.to_csv(f"{self.save_dir}/device_details.csv", index=False)

        # 4. 网络流量历史（新增）
        if self.traffic_history:
            traffic_df = pd.DataFrame({
                'round': range(1, len(self.traffic_history)+1),
                'total_traffic_bits': self.traffic_history
            })
            traffic_df.to_csv(f"{self.save_dir}/traffic_history.csv", index=False)

        # 5. 保存配置
        with open(f"{self.save_dir}/config.json", 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)

    def generate_report(self) -> str:
        report = "=" * 60 + "\n"
        report += "Edge Intelligence Simulation Report\n"
        report += "=" * 60 + "\n\n"
        report += f"Rounds: {len(self.history['round'])}\n"
        report += f"Final Accuracy: {self.history['accuracy'][-1]:.2f}%\n"
        report += f"Best Accuracy: {max(self.history['accuracy']):.2f}%\n"
        report += f"Total Energy: {sum(self.history['total_energy']):.2f} J\n"
        report += f"Total Time: {sum(self.history['total_time']):.2f} s\n"
        report += f"Average Fairness: {np.mean(self.history['fairness']):.2f}\n"
        if self.traffic_history:
            report += f"Total Traffic: {sum(self.traffic_history):.2f} bits\n"
        report += "\nConfiguration:\n"
        for k, v in self.config.to_dict().items():
            report += f"  {k}: {v}\n"
        return report