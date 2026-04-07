import torch
from models.lenet import LeNet5
from models.model_factory import get_model

class EdgeServer:
    def __init__(self, server_id, config):
        self.id = server_id
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
        self.devices = []
        self.device_updates = []
        self.total_energy = 0.0
        self.total_time = 0.0

    def add_device(self, device):
        self.devices.append(device)

    def reset_metrics(self):
        self.total_energy = 0.0
        self.total_time = 0.0

    def receive_update(self, state_dict, sample_size, device_id, device_energy, device_time):
        self.total_energy += device_energy
        self.total_time += device_time
        self.device_updates.append((state_dict, sample_size, device_id))

    def aggregate_locally(self):
        if not self.device_updates:
            return
        total_samples = sum(item[1] for item in self.device_updates)
        global_dict = self.global_model.state_dict()
        for key in global_dict:
            # 检查参数类型是否为浮点型
            if global_dict[key].dtype in (torch.float32, torch.float64, torch.float16):
                # 浮点型参数：加权平均
                global_dict[key] = torch.zeros_like(global_dict[key], dtype=torch.float32)
                for update, sample_size, _ in self.device_updates:
                    weight = sample_size / total_samples
                    update_val = update[key].float() if update[key].is_floating_point() else update[key].float()
                    global_dict[key] += weight * update_val
            else:
                # 非浮点型参数（如整型计数器）：直接取第一个设备的参数
                if self.device_updates:
                    global_dict[key] = self.device_updates[0][0][key].clone()
        self.global_model.load_state_dict(global_dict)
        self.device_updates.clear()

        # 边缘服务器自身聚合的能耗和时间
        self.total_energy += 0.1
        self.total_time += 0.01