from catlass_cppgen.op.op import OperationBase
from typing import Union, Optional, Tuple
from catlass_cppgen.common.typing import SupportedDataType, SupportedTensor
from logging import Logger
from catlass_cppgen.kernel.kernel_base import KernelBase
import numpy as np
import torch

from catlass_cppgen.catlass.layout.layout import Layout
from catlass_cppgen.common.get_accumulator_dtype import get_default_accumulator


class OpTensor:
    def __init__(self, origin_tensor: SupportedTensor):
        self.origin_tensor = origin_tensor

    def shape(self) -> Tuple[int, ...]:
        if isinstance(self.origin_tensor, np.ndarray):
            return self.origin_tensor.shape
        elif isinstance(self.origin_tensor, torch.Tensor):
            return tuple(self.origin_tensor.shape)
        else:
            raise TypeError(f"Unsupported tensor type: {type(self.origin_tensor)}")

    def ndim(self) -> int:
        if isinstance(self.origin_tensor, np.ndarray):
            return self.origin_tensor.ndim
        elif isinstance(self.origin_tensor, torch.Tensor):
            return self.origin_tensor.ndim
        else:
            raise TypeError(f"Unsupported tensor type: {type(self.origin_tensor)}")

    def dtype(self):
        pass

    def strides(self) -> Tuple[int]:
        return self.origin_tensor.strides()


class Gemm(OperationBase):
    def __init__(
        self,
        A: Optional[SupportedTensor] = None,
        B: Optional[SupportedTensor] = None,
        Bias: Optional[SupportedTensor] = None,
        C: Optional[SupportedTensor] = None,
        alpha: float = 1.0,
        beta: float = 0.0,
        element_accumulator: Optional[SupportedDataType] = None,
        element: Optional[SupportedDataType] = None,
        layout: Optional[Layout] = None,
        element_A: Optional[SupportedDataType] = None,
        element_B: Optional[SupportedDataType] = None,
        element_C: Optional[SupportedDataType] = None,
        element_D: Optional[SupportedDataType] = None,
        layout_A: Optional[Layout] = None,
        layout_B: Optional[Layout] = None,
        layout_C: Optional[Layout] = None,
        atlas_arch="a2",
    ):
        assert element is not None or all([element_A, element_B, element_C, element_D])
        assert layout is not None or all([layout_A, layout_B, layout_C])
        self.element_A = element_A or element
        self.element_B = element_B or element
        self.element_C = element_C or element
        self.element_D = element_D or element
        assert element_accumulator is not None or all(self.element_A, self.element_B)
        self.element_accumulator = element_accumulator or get_default_accumulator(
            self.element_A, self.element_B
        )
        self.layout_A = layout_A or layout
        self.layout_B = layout_B or layout
        self.layout_C = layout_C or layout
        self.atlas_arch = atlas_arch
        self.alpha = alpha
        self.beta = beta

    def can_implement(self):
        return self.alpha == 1.0 and self.beta in [0.0, 1.0]

    def get_kernel(self, A, B, Bias, C) -> KernelBase:
        pass

    def run(self, A, B, Bias, C):
        pass
