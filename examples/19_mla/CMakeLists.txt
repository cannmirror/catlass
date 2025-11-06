# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set(MLA_KERNEL_SOURCES kernel/mla_kernel.cpp)
set(MLA_TILING_SOURCES tiling/mla_tiling.cpp)
set_source_files_properties(kernel/mla_kernel.cpp PROPERTIES LANGUAGE ASC)

add_library(mla_kernel STATIC ${MLA_KERNEL_SOURCES})
add_library(mla_tiling STATIC ${MLA_TILING_SOURCES})

target_compile_options(mla_kernel PRIVATE --npu-arch=dav-2201)
target_link_libraries(mla_tiling PRIVATE m)

set(MLA_SOURCES mla.cpp)

catlass_example_add_executable(19_mla mix ${MLA_SOURCES})
target_include_directories(19_mla PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(19_mla PRIVATE mla_kernel mla_tiling)
