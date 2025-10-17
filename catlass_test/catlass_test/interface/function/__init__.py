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

import os
from typing import Iterable, List, Optional, Union

import torch
from catlass_test.common import get_example_template_src_path
from catlass_test.adapter import (
    BatchedMatmulAdapter,
    GroupedMatmulAdapter,
    MatmulAdapter,
)
from catlass_test.common import OpType


def basic_matmul(
    input: torch.Tensor,
    mat2: torch.Tensor,
    out_dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Test function for `examples/00_basic_matmul`.
    This function is equal to `torch.mm`.
    """
    output_tensors = {"C": out} if out is not None else {}
    adapter = MatmulAdapter(
        get_example_template_src_path("00_basic_matmul"),
        {"A": input, "B": mat2},
        output_tensors,
        {"out_dtype": out_dtype},
        OpType.AIC_ONLY,
    )
    adapter.run()
    return adapter.get_tensor("C")


def padding_matmul(
    input: torch.Tensor,
    mat2: torch.Tensor,
    out_dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Test function for `examples/04_padding_matmul`.
    This function is equal to `torch.mm`.
    """
    output_tensors = {"C": out} if out is not None else {}
    adapter = MatmulAdapter(
        get_example_template_src_path("04_padding_matmul"),
        {"A": input, "B": mat2},
        output_tensors,
        {"out_dtype": out_dtype},
        OpType.AIC_ONLY,
    )
    adapter.run()
    return adapter.get_tensor("C")


def splitk_matmul(
    input: torch.Tensor,
    mat2: torch.Tensor,
    out_dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Test function for `examples/09_splitk_matmul`.
    This function is equal to `torch.mm`.
    """
    output_tensors = {"C": out} if out is not None else {}
    adapter = MatmulAdapter(
        get_example_template_src_path("09_splitk_matmul"),
        {"A": input, "B": mat2},
        output_tensors,
        {"out_dtype": out_dtype},
        OpType.MIX_AIC_1_2,
    )
    adapter.run()
    return adapter.get_tensor("C")
