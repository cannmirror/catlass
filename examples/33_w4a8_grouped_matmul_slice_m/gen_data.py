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

import os
import sys
import torch
import numpy as np
import pandas as pd
from enum import Enum

CONST_16 = 16
RECORD_COUNT = 10
DATA_RANGE = (-1.0, 1.0)
WORKSPACE = os.getcwd()

os.environ["WORKSPACE"] = WORKSPACE
os.environ["ASCEND_GLOBAL_LOG_LEVEL"] = "3"
os.environ["ASCEND_SLOG_PRINT_TO_STDOUT"] = "0"


class CubeFormat(Enum):
    ND = 0
    NZ = 1
    ZN = 2
    ZZ = 3
    NN = 4
    VECTOR = 5

    def __repr__(self) -> str:
        return self.__name__

class OpParam:
    def __init__(self) -> None:
        self.b = 0
        self.g = 0  # group count
        self.m = 0
        self.k = 0
        self.n = 0
        self.transA = False
        self.transB = False
        self.enBias = False
        self.enScale = False
        self.enResidual = False
        self.layoutA = CubeFormat.ND
        self.layoutB = CubeFormat.ND
        self.layoutC = CubeFormat.ND

    def __str__(self) -> str:
        return f"Shape: ({self.b}, {self.g}, {self.m}, {self.k}, {self.n}) \n" + \
               f"Transpose: A {self.transA}, B {self.transB} \n" + \
               f"(De)Quant: Bias {self.enBias}, Scale {self.enScale}, Residual {self.enResidual} \n" + \
               f"Layout: layoutA {self.layoutA}, layoutB {self.layoutB}, layoutC {self.layoutC}"

# 生成一个row x col的二维数组，数值介于-8到7之间
def gen_data_int8(row, col):
    data = np.random.randint(-8, 8, size=(row, col), dtype=np.int8)  # todo
    # data = np.random.randint(-128, 128, size=(row, col), dtype=np.int8)
    return data

# 生成一个group × row x col的三维数组，数值介于-8到7之间
def gen_data_int8_3d(group, row, col):
    data = np.random.randint(-8, 8, size=(group, row, col), dtype=np.int8)
    return data

# 生成数据并做数据压缩
def gen_data_int4_3d(g, row, col, trans):
    # 生成原始数据，范围严格在 [-8, 7]，形状为 (g, row, col)
    data_int8_origin = gen_data_int8_3d(g, row, col)
    
    if trans:
        # 转置后两维，形状变为 (g, col, row)
        data_int8_origin = np.transpose(data_int8_origin, (0, 2, 1))
        
        # 如果行数为奇数，则补零列（针对转置后的第二维）
        data_int8 = data_int8_origin
        if row % 2 != 0:  # 注意：转置后原来的row变成了现在的col
            zero_slice = np.zeros((g, col, 1), dtype=np.int8)
            data_int8 = np.concatenate((data_int8_origin, zero_slice), axis=2)

        # 重塑为 (g, -1, 2) 进行量化
        quantized = data_int8.reshape(g, -1, 2)
        high_quantized = (quantized[:, :, 0] & 0x0F)
        low_quantized = (quantized[:, :, 1] & 0x0F) << 4
        data_int4 = low_quantized | high_quantized

        data_int4_array = np.array(data_int4, dtype=np.int8)
        return np.transpose(data_int8_origin, (0, 2, 1)), data_int4_array  # 为计算golden，int8转置回来

    else:
        data_int8 = data_int8_origin
        # 如果列数为奇数，则补零列（针对第三维）
        if col % 2 != 0:
            zero_slice = np.zeros((g, row, 1), dtype=np.int8)
            data_int8 = np.concatenate((data_int8_origin, zero_slice), axis=2)

        # 重塑为 (g, -1, 2) 进行量化
        quantized = data_int8.reshape(g, -1, 2)
        high_quantized = (quantized[:, :, 0] & 0x0F)
        low_quantized = (quantized[:, :, 1] & 0x0F) << 4
        data_int4 = low_quantized | high_quantized

        data_int4_array = np.array(data_int4, dtype=np.int8)
        return data_int8_origin, data_int4_array

# 生成grouplist，前缀和而非每段长度
def gen_group_list(group_count, total_len):
    random_group_len = np.random.randint(0, total_len+1, group_count)
    sorted_group_len = np.sort(random_group_len)
    prefix_sum = np.cumsum(sorted_group_len)
    prefix_sum[-1] = total_len
    return prefix_sum

# 生成g份不同的0~1之间的scale值，每个group基于该值做perTensor量化
def gen_scales(group_count):
    return np.random.rand(group_count).astype(np.float32)

def gen_testcase(path: str, param: OpParam) -> None:
    bsize, gcount, msize, ksize, nsize = param.b, param.g, param.m, param.k, param.n
    transA, transB = param.transA, param.transB

    group_list = gen_group_list(gcount, msize)
    group_list.tofile(os.path.join(path, "inputGroupList.dat"))
    assert group_list[-1] == msize, f"group_list最后一个元素 {group_list[-1]} 不等于 m={msize}"
    split_points = group_list[:-1]

    scales = gen_scales(gcount)
    scales.tofile(os.path.join(path, "inputScales.dat"))

    a_int8 = gen_data_int8(msize, ksize)
    a_splits = np.split(a_int8, split_points)

    b_int8, b_int4 = gen_data_int4_3d(gcount, ksize, nsize, transB)
    b_int4.tofile(os.path.join(path, "inputB.dat"))

    # numpy int32矩阵乘法非常慢，此处用float32的矩阵乘法代替
    res_splits = []
    for i in range(gcount):
        res_split = np.dot(a_splits[i].astype(np.float32), b_int8[i].astype(np.float32))
        res_split = scales[i] * res_split
        res_splits.append(res_split)
    c_float = np.vstack(res_splits)
    c_half = c_float.astype(np.float16)

    if transA:
        a_int8 = a_int8.T
    a_int8.tofile(os.path.join(path, "inputA.dat"))
    c_half.tofile(os.path.join(path, "inputC.dat"))
    c_float.tofile(os.path.join(path, "expected.dat"))

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))

    param = OpParam()
    param.b = 1
    param.g = int(sys.argv[1])
    param.m = int(sys.argv[2])
    param.n = int(sys.argv[3])
    param.k = int(sys.argv[4])
    param.transA = 0
    param.transB = 0

    data_dir = os.path.join(current_dir, "data")

    os.makedirs(data_dir, exist_ok=True)
    gen_testcase(data_dir, param)