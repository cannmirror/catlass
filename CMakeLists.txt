# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.22)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(CATLASS REQUIRED)

project(catlass LANGUAGES ASC CXX)

if(NOT DEFINED CMAKE_BUILD_TYPE OR NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

if(NOT DEFINED CATLASS_BISHENG_ARCH OR NOT CATLASS_BISHENG_ARCH)
    message(WARNING "CATLASS_BISHENG_ARCH is not defined, use default value \"a2\"")
    set(CATLASS_BISHENG_ARCH a2)
endif()

# Compile options for secure compile
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_SKIP_RPATH TRUE)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fstack-protector-strong>)
    add_compile_options("SHELL:$<$<COMPILE_LANGUAGE:ASC>:-Xhost-start -fstack-protector-strong -Xhost-end>")
    add_link_options(-Wl,-z,relro,-z,now)
    add_link_options(-Wl,-z,noexecstack)
    add_link_options(-s)
endif()

add_subdirectory(examples)
add_subdirectory(tools)

if(DEFINED BUILD_TESTS AND BUILD_TESTS)
    add_subdirectory(tests)
endif()
