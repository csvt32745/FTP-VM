import math
import torch


class MemoryBank:
    def __init__(self, top_k=5):
        self.top_k = top_k
        self.mem_k = None
        self.mem_v = None
        self.temp_k = None
        self.temp_v = None

    def memory_pruning(self):
        if self.mem_k.size(1) > self.top_k:
            self.mem_k = torch.cat([self.mem_k[:, :1], self.mem_k[:,  -(self.top_k):]], dim=1).contiguous()
            self.mem_v = torch.cat([self.mem_v[:, :, :1], self.mem_v[:, :, -(self.top_k):]], dim=2).contiguous()

    def get_memory(self):
        if self.mem_k is None:
            return None
        if self.temp_k is not None:
            mk = torch.cat([self.mem_k, self.temp_k], 1)
            mv = torch.cat([self.mem_v, self.temp_v], 2)
        else:
            mk = self.mem_k
            mv = self.mem_v
        return mk, mv

    def add_memory(self, key, value, is_temp=False):
        if self.mem_k is None:
            # First frame, just shove it in
            self.mem_k = key
            self.mem_v = value
        else:
            if is_temp:
                self.temp_k = key
                self.temp_v = value
            else:
                self.mem_k = torch.cat([self.mem_k, key], 1)
                self.mem_v = torch.cat([self.mem_v, value], 2)
        self.memory_pruning()
    
    def add_gt_memory(self, key, value):
        if self.mem_k is None:
            # First frame, just shove it in
            self.mem_k = key
            self.mem_v = value
        else:
            self.mem_k[:, [0]] = key
            self.mem_v[:, :, [0]] = value
