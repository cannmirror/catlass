#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import random
from typing import Union
import unittest

import torch
import catlass_test
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

class CatlassTest(TestCase):

    def _run_case_basic(
        self,
        m: int,
        n: int,
        k: int,
        trans_a: bool = False,
        trans_b: bool = False,
        dtype: torch.dtype = torch.float16,
    ):
        shape1 = (m, k) if not trans_a else (k, m)
        shape2 = (k, n) if not trans_b else (n, k)

        a = torch.rand(shape1, device="npu").to(dtype)
        b = torch.rand(shape2, device="npu").to(dtype)

        a = a if not trans_a else a.T
        b = b if not trans_b else b.T

        torch.npu.synchronize()
        result = catlass_test.basic_matmul(a, b, out_dtype=dtype)
        golden = torch.mm(a, b)
        torch.npu.synchronize()
        if dtype == torch.bfloat16:
            result = result.to(torch.float32)
            golden = golden.to(torch.float32)
        self.assertRtolEqual(result, golden)

    def test_basic_matmul_pybind(self):
        self._run_case_basic(2, 3, 4)

    def test_basic_matmul_pybind_cr(self):
        self._run_case_basic(2, 3, 4, trans_a=True)

    def test_basic_matmul_pybind_rc(self):
        self._run_case_basic(2, 3, 4, trans_b=True)

    def test_basic_matmul_pybind_cc(self):
        self._run_case_basic(2, 3, 4, trans_a=True, trans_b=True)

    def test_basic_matmul_pybind_bf16(self):
        self._run_case_basic(2, 3, 4, trans_a=True, trans_b=True)


if __name__ == "__main__":
    run_tests()
