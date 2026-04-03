import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional
from config import Config
from core.device import Device

sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class Logger:
    """记录训练数据，生成可视化图表，导出CSV"""
    def __init__(self, config: Config, save_dir: str = "results"):
        self.config = config
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.history = {
            "round": [],
            "accuracy": [],
            "loss": [],
            "total_energy": [],
            "total_time": [],
            "selected_devices": [],
            "fairness": []
        }
        self.device_energies = {}   # device_id -> list of energies per participation
        self.device_accuracies = {} # device_id -> list of accuracies per participation
        self.device_rounds = {}     # device_id -> list of rounds participated

    def update(self, round_idx: int, accuracy: float, loss: float,
               total_energy: float, total_time: float,
               selected_devices: List[Device]) -> None:
        self.history["round"].append(round_idx)
        self.history["accuracy"].append(accuracy)
        self.history["loss"].append(loss)
        self.history["total_energy"].append(total_energy)
        self.history["total_time"].append(total_time)
        self.history["selected_devices"].append([d.id for d in selected_devices])

        for d in selected_devices:
            if d.id not in self.device_energies:
                self.device_energies[d.id] = []
                self.device_accuracies[d.id] = []
                self.device_rounds[d.id] = []
            self.device_energies[d.id].append(d.last_energy)
            self.device_accuracies[d.id].append(d.last_accuracy)
            self.device_rounds[d.id].append(round_idx)

        # 公平性（基于所有设备参与次数方差）
        all_counts = [d.participation_count for d in selected_devices]
        if len(all_counts) > 1:
            fairness = 1.0 / (np.var(all_counts) + 1e-6)
        else:
            fairness = 1.0
        self.history["fairness"].append(fairness)

    def visualize(self):
        """生成四类核心图表"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. 精度变化折线图
        axes[0, 0].plot(self.history["round"], self.history["accuracy"], 'b-o', linewidth=2)
        axes[0, 0].set_xlabel("Round")
        axes[0, 0].set_ylabel("Accuracy (%)")
        axes[0, 0].set_title("Global Model Accuracy")
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 设备总能耗对比图
        total_energies = [sum(energies) for energies in self.device_energies.values()]
        device_ids = list(self.device_energies.keys())
        axes[0, 1].bar(device_ids, total_energies, color='skyblue', edgecolor='black')
        axes[0, 1].set_xlabel("Device ID")
        axes[0, 1].set_ylabel("Total Energy (J)")
        axes[0, 1].set_title("Total Energy Consumption per Device")
        axes[0, 1].grid(True, alpha=0.3, axis='y')

        # 3. 每轮总时间趋势
        axes[1, 0].plot(self.history["round"], self.history["total_time"], 'r-s', linewidth=2)
        axes[1, 0].set_xlabel("Round")
        axes[1, 0].set_ylabel("Total Time (s)")
        axes[1, 0].set_title("Total Training Time per Round")
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 能耗-准确率散点图（最后一轮）
        if self.device_energies:
            last_round_energies = []
            last_round_accuracies = []
            for d_id in device_ids:
                if len(self.device_energies[d_id]) > 0:
                    last_round_energies.append(self.device_energies[d_id][-1])
                    last_round_accuracies.append(self.device_accuracies[d_id][-1])
            axes[1, 1].scatter(last_round_energies, last_round_accuracies, c=range(len(last_round_energies)), cmap='viridis', s=100)
            axes[1, 1].set_xlabel("Energy (J)")
            axes[1, 1].set_ylabel("Accuracy (%)")
            axes[1, 1].set_title("Energy-Accuracy Trade-off (Last Round)")
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/simulation_visualization.png", dpi=300, bbox_inches='tight')
        plt.show()

    def save_data(self):
        """保存历史数据为CSV，并保存配置"""
        # 保存训练结果表
        df = pd.DataFrame({
            'round': self.history["round"],
            'accuracy': self.history["accuracy"],
            'loss': self.history["loss"],
            'total_energy': self.history["total_energy"],
            'total_time': self.history["total_time"],
            'fairness': self.history["fairness"]
        })
        df.to_csv(f"{self.save_dir}/training_results.csv", index=False)

        # 保存设备能耗明细（按轮次填充，缺失值NaN）
        n_rounds = len(self.history["round"])
        device_energy_df = pd.DataFrame(index=range(1, n_rounds+1), columns=list(self.device_energies.keys()))
        device_energy_df = device_energy_df.fillna(np.nan)
        for d_id, rounds in self.device_rounds.items():
            energies = self.device_energies[d_id]
            for r, e in zip(rounds, energies):
                device_energy_df.loc[r, d_id] = e
        device_energy_df.to_csv(f"{self.save_dir}/device_energies.csv")

        # 保存配置
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
        report += f"Average Fairness: {np.mean(self.history['fairness']):.2f}\n\n"
        report += "Configuration:\n"
        for k, v in self.config.to_dict().items():
            report += f"  {k}: {v}\n"
        return report