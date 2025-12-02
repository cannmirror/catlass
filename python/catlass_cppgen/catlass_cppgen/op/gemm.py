from catlass_cppgen.op.op import OperationBase
from typing import Type, Union, Optional, Tuple
from catlass_cppgen.common.typing import SupportedDataType, SupportedTensor
from logging import Logger

from catlass_cppgen.catlass.layout.layout import Layout
from catlass_cppgen.common.data_type import get_default_accumulator
from catlass_cppgen.kernel.gemm.basic_matmul import BasicMatmulKernel
from catlass_cppgen.catlass.arch.arch import ArchTag
from catlass_cppgen.kernel.gemm.gemm_base import GemmKernelBase


class Gemm(OperationBase):
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.0,
        element_accumulator: Optional[SupportedDataType] = None,
        element: Optional[SupportedDataType] = None,
        layout: Optional[Type[Layout]] = None,
        element_A: Optional[SupportedDataType] = None,
        element_B: Optional[SupportedDataType] = None,
        element_Bias: Optional[SupportedDataType] = None,
        element_C: Optional[SupportedDataType] = None,
        layout_A: Optional[Type[Layout]] = None,
        layout_B: Optional[Type[Layout]] = None,
        layout_Bias: Optional[Type[Layout]] = None,
        epilogue=None,
        atlas_arch: Optional[ArchTag] = None,
    ):
        assert element is not None or all(
            [element_A, element_B, element_Bias, element_C]
        )
        assert layout is not None or all([layout_A, layout_B, layout_Bias])
        self.element_A = element_A or element
        self.element_B = element_B or element
        self.element_Bias = element_Bias or element
        self.element_C = element_C or element
        assert element_accumulator is not None or all([self.element_A, self.element_B])
        self.element_accumulator = element_accumulator or get_default_accumulator(
            self.element_A, self.element_B
        )
        self.layout_A = layout_A or layout
        self.layout_B = layout_B or layout
        self.layout_Bias = layout_Bias or layout_B
        self.atlas_arch = atlas_arch
        self.alpha = alpha
        self.beta = beta

    def can_implement(self):
        return self.alpha == 1.0 and self.beta in [0.0, 1.0]

    # def _check_tensor(self, tensor: OpTensor, element: SupportedDataType, layout: LayoutEnum) -> bool:
    #     if isinstance(tensor.origin_tensor, np.ndarray):
    #         return tensor.origin_tensor.dtype == element
    #     elif isinstance(tensor.origin_tensor, torch.Tensor):
    #         return tensor.origin_tensor.dtype == element
    #     else:
    #         raise TypeError(f"Unsupported tensor type: {type(tensor.origin_tensor)}")

    def get_kernel(
        self,
        A: Optional[SupportedTensor] = None,
        B: Optional[SupportedTensor] = None,
        Bias: Optional[SupportedTensor] = None,
        C: Optional[SupportedTensor] = None,
    ) -> Optional[GemmKernelBase]:
        params = {
            "A": A,
            "B": B,
            "Bias": Bias,
            "C": C,
            "element_accumulator": self.element_accumulator,
            "element_A": self.element_A,
            "element_B": self.element_B,
            "element_Bias": self.element_Bias,
            "element_C": self.element_C,
            "layout_A": self.layout_A,
            "layout_B": self.layout_B,
            "layout_Bias": self.layout_Bias,
            "arch_tag": self.atlas_arch,
        }

        # assert self._check_tensor(A, self.element_A, self.layout_A)
        # assert self._check_tensor(B, self.element_B, self.layout_B)
        # assert self._check_tensor(Bias, self.element_Bias, self.layout_Bias)
        # assert self._check_tensor(C, self.element_C, self.layout_C)

        if self.alpha == 1.0 and self.beta == 0.0:
            if Bias is not None:
                pass
            return BasicMatmulKernel(**params)
        elif self.alpha == 1.0 and self.beta == 1.0:
            pass
        return None

    def run(self, A, B, Bias, C):
        pass
