import torch
from . import modules
from . import ops
from typing import List, Optional

__all__ = [
    "QTensor", 
    "TensorPack",  
]

class QTensor:
    def __init__(self, quantized_x: torch.Tensor, scales_x: torch.Tensor):
        self.quantized_x = quantized_x
        self.scales_x = scales_x

    def size(self):
        return self.quantized_x.size()

    @property
    def device(self):
        return self.quantized_x.device

    @property
    def dtype(self):
        return self.quantized_x.dtype


class TensorPack:

    def __init__(
        self,
        x: torch.Tensor | QTensor,
        outlier_x: Optional[torch.Tensor] = None,
    ):
        self.x = x
        self.outlier_x = outlier_x

    @property
    def device(self):
        return self.x.device

    @property
    def dtype(self):
        return self.x.dtype
