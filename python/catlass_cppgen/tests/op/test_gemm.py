import unittest
from catlass_cppgen.op.gemm import Gemm
import random

from catlass_cppgen.common.data_type import DataType


class TestGemmCoord(unittest.TestCase):
    def test_gemm_codegen(self):
        gemm_plan = Gemm(element=DataType.FLOAT32, layout=RowMajor)


if __name__ == "__main__":
    unittest.main()
