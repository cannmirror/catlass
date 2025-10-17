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
from catlass_test import (
    ASCEND_HOME_PATH,
    CATLASS_TEST_PATH,
    CATLASS_TEST_INCLUDE_PATH,
    CATLASS_INCLUDE_PATH,
)

CATLASS_KERNEL_ENTRY_FILE = os.path.join(CATLASS_TEST_PATH, "csrc", "kernel.cpp")

COMPILER_INCLUDE_DIRECTORIES = [
    f"-I{CATLASS_TEST_INCLUDE_PATH}",
    f"-I{ASCEND_HOME_PATH}/include",
    f"-I{ASCEND_HOME_PATH}/include/experiment/runtime",
    f"-I{ASCEND_HOME_PATH}/include/experiment/msprof",
    f"-I{ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw",
    f"-I{ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl",
    f"-I{ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface",
    f"-I{CATLASS_INCLUDE_PATH}",
]

COMPILER_COMPILE_OPTIONS = [
    "-xcce",
    "-std=c++17",
    "-Wno-macro-redefined",
]
COMPILER_LLVM_COMPILE_OPTIONS = [
    "--cce-aicore-arch=dav-c220",
    "-mllvm",
    "-cce-aicore-stack-size=0x8000",
    "-mllvm",
    "-cce-aicore-function-stack-size=0x8000",
    "-mllvm",
    "-cce-aicore-record-overflow=true",
    "-mllvm",
    "-cce-aicore-addr-transform",
    "-mllvm",
    "-cce-aicore-dcci-insert-for-scalar=false",
]




COMPILER_LINK_DIRECTORIES = [
    f"-L{ASCEND_HOME_PATH}/lib64",
]
COMPILER_LINK_LIBRARIES = [
    "-ltiling_api",
    "-lascendcl",
    "-lstdc++",
]
