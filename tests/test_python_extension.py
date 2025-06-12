# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from torch_npu.testing.testcase import TestCase, run_tests
import torch_npu
import torch
import torch_catlass
import random


def generate_sequence_split(n, S):
    split_points = sorted([random.randint(0, S) for _ in range(n - 1)])
    split_points = [0] + split_points + [S]
    sequence = [split_points[i+1] - split_points[i] for i in range(n)]
    sequence[-1] = S - sum(sequence[:-1])
    return sequence


def calculate_prefix_sum(sequence):
    prefix_sum = []
    current_sum = 0
    for num in sequence:
        current_sum += num
        prefix_sum.append(current_sum)
    return prefix_sum


class CatlassTest(TestCase):

    def _run_case_basic(self, m: int, n: int, k: int, transA: bool = False, transB: bool = False, dtype: torch.dtype = torch.float16):
        shape1 = (m, k) if not transA else (k, m)
        shape2 = (k, n) if not transB else (n, k)

        a = torch.rand(shape1, device='npu').to(dtype)
        b = torch.rand(shape2, device='npu').to(dtype)

        a = a if not transA else a.T
        b = b if not transB else b.T

        torch.npu.synchronize()
        result = torch_catlass.basic_matmul(a, b, str(dtype).split('.')[-1])
        golden = torch.mm(a, b)
        torch.npu.synchronize()
        if dtype == torch.bfloat16:
            result = result.to(torch.float32)
            golden = golden.to(torch.float32)
        self.assertRtolEqual(result, golden)

    def test_basic_matmul_pybind(self):
        self._run_case_basic(2, 3, 4)

    def test_basic_matmul_pybind_cr(self):
        self._run_case_basic(2, 3, 4, transA=True)

    def test_basic_matmul_pybind_rc(self):
        self._run_case_basic(2, 3, 4, transB=True)

    def test_basic_matmul_pybind_cc(self):
        self._run_case_basic(2, 3, 4, transA=True, transB=True)

    def test_basic_matmul_pybind_bf16(self):
        self._run_case_basic(2, 3, 4, transA=True, transB=True)

    def test_grouped_matmul_list_m(self):
        g = 128
        groupList = generate_sequence_split(g, random.randint(256, 40960))
        groupList = calculate_prefix_sum(groupList)
        M, k, n = sum(groupList), 4096, 1280
        a = torch.randn((M, k), device='npu').to(torch.float16)
        b = torch.randn((g, k, n), device='npu').to(torch.float16)
        b_list = [b[i] for i in range(g)]
        groupListTensor = torch.tensor(groupList, device='npu').to(torch.int64)
        # input, weight, group_list, dtype, transpose_a, transpose_b, 是否为切K
        result = torch_catlass.grouped_matmul(
            a, b, groupListTensor, "float16", False, False, False)
        golden = torch_npu.npu_grouped_matmul(
            [a], b_list, group_list=groupList, split_item=3)[0]
        self.assertRtolEqual(result, golden)

    def test_grouped_matmul_list_k(self):
        g = 128
        groupList = generate_sequence_split(g, random.randint(256, 40960))
        groupList = calculate_prefix_sum(groupList)
        m, K, n = 4096, sum(groupList), 1280
        a = torch.randn((K, m), device='npu').to(torch.float16)
        b = torch.randn((K, n), device='npu').to(torch.float16)
        groupListTensor = torch.tensor(groupList, device='npu').to(torch.int64)
        a_list = torch.split(a.transpose(0, 1).contiguous(), groupList, dim=1)
        b_list = torch.split(b, groupList, dim=0)
        result = torch_catlass.grouped_matmul(
            a, b, groupListTensor, "float16", True, False, True)
        golden = torch.stack(torch_npu.npu_grouped_matmul(a_list, b_list))
        self.assertRtolEqual(result, golden)

    def test_optimized_matmul_pybind(self):
        a = torch.rand((2, 3), device='npu').to(torch.float16)
        b = torch.rand((3, 4), device='npu').to(torch.float16)
        torch.npu.synchronize()
        result = torch_catlass.optimized_matmul(a, b, "float16")
        golden = torch.mm(a, b)
        torch.npu.synchronize()
        self.assertRtolEqual(result, golden)


if __name__ == "__main__":
    run_tests()
