import torch

class Config:
    """全局配置类"""
    def __init__(self):
        # 联邦学习基本配置
        self.base_station_mode = "single"  # 或 "multi"，默认单基站
        self.num_edge_servers = 2  # 边缘服务器数量（默认2）
        self.num_devices = 5          # 边缘设备数量（5~10台）
        self.num_rounds = 5          # 联邦学习轮次
        self.local_epochs = 3         # 本地训练轮数
        self.batch_size = 64          # 本地训练批次大小
        self.learning_rate = 0.01     # 学习率
        self.model_size_bits = 2.5e6  # 模型大小（比特），LeNet-5约2.5M比特
        self.gpu_ratio = 0.5  # GPU 设备所占比例（0.5 表示一半 GPU，一半 CPU）

        # 设备选择策略
        self.selection_ratio = 0.5     # 每轮选择的设备比例
        self.selection_strategy = "hybrid"  # random, energy_aware, capability_aware, hybrid

        # 聚合算法
        self.aggregation_method = "fedavg"  # fedavg, fedprox, fedavg_energy, qfed, capability_weighted

        # 设备异构性配置
        self.cpu_freq_range = (1.0, 3.0)      # GHz
        self.gpu_freq_range = (0.8, 1.8)      # GHz
        self.bandwidth_range = (1e5, 5e5)     # bits/s
        self.power_range = (0.5, 1.5)         # W
        self.compute_capacity_range = (0.5, 1.5)  # 计算能力因子

        # 动态波动幅度（每轮变化比例）
        self.dynamic_fluctuation = 0.1        # ±10%

        # 能耗模型参数（范围，GPU取较低值，CPU取较高值）
        self.k_cpu_range = (1e-28, 1e-27)     # CPU能耗系数范围
        self.k_gpu_range = (1e-28, 5e-28)     # GPU能耗系数范围
        self.energy_weight = 0.7      # 能量效率权重
        self.accuracy_weight = 0.3    # 准确率权重

        # 数据异构性
        self.data_iid_ratio = 0.8     # IID数据比例
        self.class_imbalance_factor = 0.3  # 非IID偏斜程度

        # 实验设置
        self.seed = 42
        self.use_cuda = torch.cuda.is_available()
        self.save_checkpoints = True
        self.visualize = True

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}