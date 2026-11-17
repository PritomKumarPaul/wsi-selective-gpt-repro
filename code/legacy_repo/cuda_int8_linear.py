"""Dependency-free W8A8 Linear using PyTorch 2.5 native CUDA INT8 GEMM.

Symmetric per-output-channel weights and per-input-row activation scales.
Only Linear operators are converted; other operators retain their original
dtype. This uses the private torch._int_mm API and fails if it is unavailable.
Conversion overhead and non-Linear operations mean speedup is not guaranteed.
"""
import torch
import torch.nn.functional as F
from torch import nn


class CudaInt8Linear(nn.Module):
    def __init__(self, layer):
        super().__init__()
        weight = layer.weight.detach().float()
        self.in_features = layer.in_features
        self.out_features = layer.out_features
        padded_in = (self.in_features + 31) // 32 * 32
        padded_out = (self.out_features + 7) // 8 * 8
        scale = weight.abs().amax(dim=1).clamp_min(1e-8) / 127
        quantized = torch.round(weight / scale[:, None]).clamp(-127, 127).to(torch.int8)
        quantized = F.pad(quantized, (0, padded_in-self.in_features, 0, padded_out-self.out_features))
        self.register_buffer('weight_t', quantized.t().contiguous())
        self.register_buffer('weight_scale', scale)
        self.register_buffer('bias', layer.bias.detach().float().clone() if layer.bias is not None else None)

    def forward(self, x):
        if not x.is_cuda:
            raise RuntimeError('CudaInt8Linear requires CUDA; it never silently falls back to FP32')
        shape = x.shape
        matrix = x.reshape(-1, self.in_features).float()
        scale = matrix.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127
        q = torch.round(matrix / scale).clamp(-127, 127).to(torch.int8)
        # CUDA _int_mm in this installed build requires M>16, K divisible by32,
        # N divisible by8; padding also covers singleton attention queries.
        padded_rows = max(32, (len(matrix)+31)//32*32)
        q = F.pad(q, (0, self.weight_t.shape[0]-self.in_features, 0, padded_rows-len(matrix))).contiguous()
        accum = torch._int_mm(q, self.weight_t)
        result = accum[:len(matrix), :self.out_features].float() * scale * self.weight_scale[None, :]
        if self.bias is not None:
            result += self.bias
        return result.reshape(*shape[:-1], self.out_features).to(x.dtype)


def convert_cuda_linears(module):
    count = 0
    for name, child in list(module.named_children()):
        if type(child) is nn.Linear:
            setattr(module, name, CudaInt8Linear(child))
            count += 1
        else:
            count += convert_cuda_linears(child)
    return count


def check_cuda_int8():
    if not torch.cuda.is_available() or not hasattr(torch, '_int_mm'):
        raise RuntimeError('This run requires CUDA and torch._int_mm')
    a = torch.ones((32,32),dtype=torch.int8,device='cuda')
    b = torch.ones((32,8),dtype=torch.int8,device='cuda')
    if not torch.equal(torch._int_mm(a,b), torch.full((32,8),32,dtype=torch.int32,device='cuda')):
        raise RuntimeError('Native CUDA INT8 GEMM self-check failed')
