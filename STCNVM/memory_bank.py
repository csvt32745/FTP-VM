import math
import torch


def softmax_w_top(x, top):
    values, indices = torch.topk(x, k=top, dim=1)
    # x_exp = values.exp_()
    x_exp = torch.exp(values)
    # print(x_exp.max(), x_exp.min())
    x_exp /= torch.sum(x_exp, dim=1, keepdim=True)
    # The types should be the same already
    # some people report an error here so an additional guard is added
    x.zero_().scatter_(1, indices, x_exp.type(x.dtype)) # B * THW * HW

    return x


class MemoryBank:
    def __init__(self, k=1, top_k=20):
        self.top_k = top_k

        self.CK = None
        self.CV = None
        self.SK = None
        self.SV = None

        self.mem_k = None
        self.mem_v = None
        self.mem_timestamp = []

        self.num_objects = k

        self.mem_count = 0

    def memory_pruning(self):
        frame_count = self.mem_k.size(2) // self.SK
        if frame_count > self.top_k:
            self.mem_k = torch.cat([self.mem_k[..., :self.SK], self.mem_k[..., -(self.SK*self.top_k):]], dim=2).contiguous()
            self.mem_v = torch.cat([self.mem_v[..., :self.SV], self.mem_v[..., -(self.SV*self.top_k):]], dim=2).contiguous()

    def _global_matching(self, mk, qk):
        # NE means number of elements -- typically T*H*W
        B, CK, NE = mk.shape

        # See supplementary material
        a_sq = mk.pow(2).sum(1).unsqueeze(2)
        ab = mk.transpose(1, 2) @ qk
        affinity = (2*ab-a_sq) / math.sqrt(CK)   # B, NE, HW
        # affinity = softmax_w_top(affinity, top=self.top_k)  # B, NE, HW
        affinity = torch.nn.functional.softmax(affinity, dim=1)
        return affinity

    def _readout(self, affinity, mv):
        return torch.bmm(mv, affinity)

    def get_memory(self):
        if self.temp_k is not None:
            mk = torch.cat([self.mem_k, self.temp_k], 2)
            mv = torch.cat([self.mem_v, self.temp_v], 2)
        else:
            mk = self.mem_k
            mv = self.mem_v
        return mk, mv

    def match_memory(self, qk):
        k = self.num_objects
        b, c, t, h, w = qk.shape
        qk = qk.flatten(start_dim=2) # B C (T H W)
        mk, mv = self.get_memory()
        # print(qk.isnan().any(), mk.isnan().any(), mv.isnan().any())
        affinity = self._global_matching(mk, qk)
        # print(affinity.isnan().any())
        # One affinity for all
        readout_mem = self._readout(affinity, mv)
        # print(readout_mem.isnan().any())
        # readout_mem = self._readout(affinity.expand(k,-1,-1), mv)

        return readout_mem.view(b, t, self.CV, h, w)

    def add_memory(self, key, value, is_temp=False):
        # Temp is for "last frame"
        # Not always used
        # But can always be flushed
        self.temp_k = None
        self.temp_v = None
        key = key.flatten(start_dim=2)
        value = value.flatten(start_dim=2)

        if self.mem_k is None:
            # First frame, just shove it in
            self.mem_k = key
            self.mem_v = value
            self.CK, self.SK = key.shape[1:]
            self.CV, self.SV = value.shape[1:]
        else:
            if is_temp:
                self.temp_k = key
                self.temp_v = value
            else:
                self.mem_k = torch.cat([self.mem_k, key], 2)
                self.mem_v = torch.cat([self.mem_v, value], 2)