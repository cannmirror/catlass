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

from typing import Dict, List, Literal, Tuple

from catlass_test.adapter import AdapterBase
from catlass_test.catlass.gemm_coord import GemmCoord
from catlass_test.common import OpType


class MatmulAdapter(AdapterBase):
    def __init__(
        self,
        kernel_src_file,
        input_tensors,
        output_tensors=...,
        attrs=...,
        op_type=OpType.MIX_AIC_1_2,
        features: List[str] = [],
    ):
        super().__init__(
            kernel_src_file, input_tensors, output_tensors, attrs, op_type, features
        )

    def __get_output_name(self):
        return "C" if "dequant" not in self.features else "D"

    @property
    def Output(self):
        return self.get_tensor(self.__get_output_name())

    def check(self):
        pass

    def __check_tensor(self):
        assert "A" in self.input_tensors
        assert "B" in self.input_tensors
        assert "mmad" not in self.features or "Bias" in self.input_tensors
        assert "dequant" not in self.features or (
            "Scale" in self.input_tensors and "PerTokenScale" in self.input_tensors
        )
        assert self.__get_output_name(self) in self.output_tensors

    def __check_tensor_dims(self):
        assert len(self.A.shape) >= 2
        assert len(self.B.shape) >= 2
        assert len(self.Output.shape) >= 2
        ma, ka = self.A.shape[-2:]
        kb, nb = self.B.shape[-2:]
        mo, no = self.Output.shape[-2:]
        assert ma == mo
        assert ka == kb
        assert nb == no
        if "dequant" in self.features:
            assert len(self.Scale.shape) == 1
            assert len(self.PerTokenScale.shape == 1)
            nscale = self.Scale.shape[0]
            mper_token_scale = self.PerTokenScale.shape[0]
            assert nb == nscale
            assert ma == mper_token_scale
        if "mmad" in self.features:
            assert len(self.Bias.shape) == 1
            nbias = self.Bias.shape[0]
            assert nb == nbias

    def __check_tensor_dtypes(self):
        dtype_a = self.A.dtype
        dtype_b = self.B.dtype
        assert dtype_a == dtype_b
        dtype_output = self.Output.dtype
        if "mmad" in self.features:
            pass
            

    def get_runtime_params(self, param_name: str):
        if param_name == "problemShape":
            return self.get_problem_shape()
        elif param_name == "problemCount":
            return self.get_problem_count()

    def get_output_shapes(self) -> Dict[str, Tuple[int, ...]]:
        ma, ka = self.A.shape[-2:]
        kb, nb = self.B.shape[-2:]
        assert ka == kb
        return {"C": (ma, nb)}

    def get_output_dtypes(self):
        if self.attrs.get("output_dtype") is None:
            assert self.A.dtype == self.B.dtype
            self.attrs["output_dtype"] = self.A.dtype
        return {"C": self.attrs["output_dtype"]}

    def get_problem_shape(self) -> GemmCoord:
        assert "A" in self.input_tensors
        assert "B" in self.input_tensors
        assert "C" in self.output_tensors
        # ma, ka = (swap(*self.A.shape) if self.get_transpose("A") else self.A.shape)[-2:]
        # kb, nb = (swap(*self.B.shape) if self.get_transpose("B") else self.B.shape)[-2:]
        ma, ka = self.A.shape[-2:]
        kb, nb = self.B.shape[-2:]
        mc, nc = self.C.shape[-2:]
        assert ma == mc
        assert ka == kb
        assert nb == nc

        return GemmCoord(ma, nb, ka)

    def get_problem_count(self) -> int:
        return 1

    @property
    def A(self):
        return self.get_tensor("A")

    @property
    def B(self):
        return self.get_tensor("B")

    @property
    def C(self):
        return self.get_tensor("C")
    
    @property
    def Bias(self):
        return self.get_tensor("Bias")

    @property
    def D(self):
        return self.get_tensor("D")

    @property
    def Scale(self):
        return self.get_tensor("Scale")

    @property
    def PerTokenScale(self):
        return self.get_tensor("PerTokenScale")

    def get_transpose(self, tensor_name: str) -> bool:
        return bool(self.attrs.get(f"Trans{tensor_name}", False))
