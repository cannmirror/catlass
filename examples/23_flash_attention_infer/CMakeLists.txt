# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set(FAI_KERNEL_SOURCES kernel/fai_kernel.cpp)
set(FAI_TILING_SOURCES tiling/fai_tiling.cpp)
set_source_files_properties(kernel/fai_kernel.cpp PROPERTIES LANGUAGE ASC)

add_library(fai_kernel STATIC ${FAI_KERNEL_SOURCES})
add_library(fai_tiling STATIC ${FAI_TILING_SOURCES})

target_compile_options(fai_kernel PRIVATE --npu-arch=dav-2201)
target_link_libraries(fai_tiling PRIVATE m)

set(FAI_SOURCES fai.cpp)

catlass_example_add_executable(23_flash_attention_infer mix ${FAI_SOURCES})
target_include_directories(23_flash_attention_infer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(23_flash_attention_infer PRIVATE fai_kernel fai_tiling)
