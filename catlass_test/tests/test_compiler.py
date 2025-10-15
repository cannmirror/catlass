import os
import unittest

from catlass_test.compiler import TemplateCompiler, CATLASS_TEST_KERNEL_PATH
from catlass_test.common import OpType
from catlass_test import CATLASS_TEST_KERNEL_EXAMPLES_PATH


class TestCompiler(unittest.TestCase):
    def test_compiler_template(self):
        t = TemplateCompiler(
            os.path.join(
                CATLASS_TEST_KERNEL_EXAMPLES_PATH, "00_basic_matmul", "basic_matmul.hpp"
            ),
        )
        for v in ["A", "B", "C"]:
            self.assertEqual(t.compile_params.get(f"Element{v}"), "class")
            self.assertEqual(t.compile_params.get(f"Layout{v}"), "class")
            self.assertEqual(t.runtime_params.get(f"device{v}"), "uint8_t*")
        self.assertEqual(t.runtime_params.get("problemShape"), "GemmCoord")
        self.assertEqual(t.runtime_params.get("stream"), "aclrtStream")

    def test_compiler_compile(self):
        if os.path.exists(CATLASS_TEST_KERNEL_PATH):
            os.system(f"rm -rf {CATLASS_TEST_KERNEL_PATH}/*")
        t = TemplateCompiler(
            os.path.join(
                CATLASS_TEST_KERNEL_EXAMPLES_PATH, "00_basic_matmul", "basic_matmul.hpp"
            ),
        )
        kernel_name = t.compile(
            {
                "ElementA": "half",
                "ElementB": "half",
                "ElementC": "half",
                "LayoutA": "layout::RowMajor",
                "LayoutB": "layout::RowMajor",
                "LayoutC": "layout::RowMajor",
            },
            op_type=OpType.MIX_AIC_1_1,
        )
        self.assertEqual(
            kernel_name,
            os.path.join(
                CATLASS_TEST_KERNEL_PATH,
                "libbasic_matmul_half_rowmajor_half_rowmajor_half_rowmajor.so",
            ),
        )


if __name__ == "__main__":
    unittest.main()
