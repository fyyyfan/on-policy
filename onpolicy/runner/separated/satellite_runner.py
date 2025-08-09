import time
import numpy as np
import torch
from onpolicy.runner.separated.base_runner import Runner
import wandb
import imageio

def _t2n(x):
    return x.detach().cpu().numpy()

class SatelliteRunner(Runner):
    """Separated Runner class for Satellite environment. 
    Currently delegates to shared implementation since satellite environment 
    is designed for shared policy training."""
    
    def __init__(self, config):
        super(SatelliteRunner, self).__init__(config)
        # 导入shared版本的实现
        from onpolicy.runner.shared.satellite_runner import SatelliteRunner as SharedSatelliteRunner
        self.shared_runner = SharedSatelliteRunner(config)

    def run(self):
        """Delegate to shared runner implementation"""
        return self.shared_runner.run()

    def warmup(self):
        """Delegate to shared runner implementation"""
        return self.shared_runner.warmup()

    def collect(self, step):
        """Delegate to shared runner implementation"""
        return self.shared_runner.collect(step)

    def insert(self, data):
        """Delegate to shared runner implementation"""
        return self.shared_runner.insert(data)

    def compute(self):
        """Delegate to shared runner implementation"""
        return self.shared_runner.compute()

    def train(self):
        """Delegate to shared runner implementation"""
        return self.shared_runner.train()

    def save(self, episode=0):
        """Delegate to shared runner implementation"""
        return self.shared_runner.save(episode)

    def restore(self, model_dir):
        """Delegate to shared runner implementation"""
        return self.shared_runner.restore(model_dir)

    def log_train(self, train_infos, total_num_steps):
        """Delegate to shared runner implementation"""
        return self.shared_runner.log_train(train_infos, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        """Delegate to shared runner implementation"""
        return self.shared_runner.log_env(env_infos, total_num_steps)

    def eval(self, total_num_steps):
        """Delegate to shared runner implementation"""
        return self.shared_runner.eval(total_num_steps)

    def render(self):
        """Delegate to shared runner implementation"""
        return self.shared_runner.render()

