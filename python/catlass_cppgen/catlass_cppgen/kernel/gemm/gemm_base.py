from typing import Any, Optional, Tuple

from catlass_cppgen.catlass.layout.layout import Layout
from catlass_cppgen.kernel.kernel_base import KernelBase
from catlass_cppgen.common.data_type import DataType
from catlass_cppgen.catlass.arch.arch import ArchTag
from catlass_cppgen.catlass.gemm_coord import GemmCoord


class GemmKernelBase(KernelBase):
    def __init__(
        self,
        element_accumulator: DataType,
        element_A: DataType,
        element_B: DataType,
        element_C: DataType,
        element_Bias: DataType,
        layout_A: Layout,
        layout_B: Layout,
        layout_Bias: Layout,
        arch_tag: ArchTag,
        *args, **kwargs
    ):
        self.element_A = element_A
        self.element_B = element_B
        self.element_Bias = element_Bias
        self.element_C = element_C
        self.layout_A = layout_A
        self.layout_B = layout_B
        self.layout_Bias = layout_Bias
        self.element_accumulator = element_accumulator
        self.arch_tag = arch_tag

    def codegen(self) -> str:
        params = {
            "arch_tag": self.arch_tag,
            "l1_tile_shape": self.l1_tile_shape,
            "l0_tile_shape": self.l0_tile_shape,
            "element_A": self.element_A,
            "element_B": self.element_B,
            "element_Bias": self.element_Bias,
            "element_C": self.element_C,
            "layout_A": self.layout_A,
            "layout_B": self.layout_B,
            "layout_Bias": self.layout_Bias,
        }
        return super().codegen(params)
