import unittest
from catlass_cppgen.catlass.gemm_coord import GemmShape
import random


class TestGemmCoord(unittest.TestCase):
    def test_gemm_shape_str(self):
        m, n, k = [random.randint(0, 512)] * 3
        gemm_shape = GemmShape(m, n, k)
        print(gemm_shape.__str__())
        self.assertEqual(str(gemm_shape), f"GemmShape<{m}, {n}, {k}>")


if __name__ == "__main__":
    unittest.main()
