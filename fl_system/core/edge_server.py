import torch
from models.lenet import LeNet5

class EdgeServer:
    def __init__(self, server_id, config):
        self.id = server_id
        self.config = config
        self.global_model = LeNet5()
        if config.use_cuda:
            self.global_model = self.global_model.cuda()
        self.devices = []          # 管辖的设备列表
        self.device_updates = []   # 存储 (state_dict, sample_size, device_id)
        # 新增：能耗时间累计
        self.total_energy = 0.0
        self.total_time = 0.0

    def add_device(self, device):
        self.devices.append(device)

    def reset_metrics(self):
        """每轮开始前调用，清零累计能耗和时间"""
        self.total_energy = 0.0
        self.total_time = 0.0

    def receive_update(self, state_dict, sample_size, device_id, device_energy, device_time):
        """接收设备更新，同时累加该设备的能耗和时间"""
        self.total_energy += device_energy
        self.total_time += device_time
        self.device_updates.append((state_dict, sample_size, device_id))

    def aggregate_locally(self):
        """聚合所辖设备的更新，并加上边缘服务器自身的聚合能耗"""
        if not self.device_updates:
            return
        total_samples = sum(item[1] for item in self.device_updates)
        global_dict = self.global_model.state_dict()
        for key in global_dict:
            global_dict[key] = torch.zeros_like(global_dict[key])
            for update, sample_size, _ in self.device_updates:
                weight = sample_size / total_samples
                global_dict[key] += weight * update[key]
        self.global_model.load_state_dict(global_dict)
        self.device_updates.clear()

        # 边缘服务器自身聚合的能耗和时间（可根据需要调整）
        self.total_energy += 0.1      # 假设聚合能耗0.1焦耳
        self.total_time += 0.01       # 假设聚合时间0.01秒