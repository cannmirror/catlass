# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.15)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
list(APPEND CMAKE_MODULE_PATH ${ASCEND_CMAKE_PATH})
project(Catlass LANGUAGES ASCEND CXX)

if(NOT DEFINED KERNEL_TEMPLATE_FILE)
message("Kernel template file is not defined")
endif()

if(NOT DEFINED KERNEL_TEMPLATE_NAME)
message("Kernel template name is not defined")

#define KERNEL_TEMPLATE_NAME Kernel
#define COMPILE_PARAM int16_t
#define RUNTIME_PARAM uint8_t *deviceA
#define RUNTIME_PARAM_CALL deviceA

set_source_files_properties(kernel.cpp PROPERTIES LANGUAGE ASCEND)
add_library(kernel SHARED kernel.cpp)