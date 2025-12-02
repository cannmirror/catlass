import unittest
from catlass_cppgen.op.gemm import Gemm

from catlass_cppgen.common.data_type import DataType
from catlass_cppgen.catlass.layout.layout import RowMajor
from catlass_cppgen.catlass.gemm_coord import GemmShape
import torch


class TestGemmCoord(unittest.TestCase):
    def test_gemm_codegen(self):
        a = torch.ones(128, 256)
        b = torch.ones(256, 384)

        gemm_plan = Gemm(element=DataType.FLOAT, layout=RowMajor)
        gemm_kernel = gemm_plan.get_kernel(A=a, B=b)
        print(gemm_kernel.get_args())
        gemm_kernel.tune(GemmShape(128, 256, 64), GemmShape(128, 256, 64))
        gemm_kernel_src = gemm_kernel.codegen()
        print(gemm_kernel_src)


if __name__ == "__main__":
    unittest.main()
