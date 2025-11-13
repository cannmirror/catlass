# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "Cannot find ASCEND_HOME_PATH, please run set_env.sh.")
else()
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
endif()

set(ASCEND_CMAKE_PATH ${CMAKE_CURRENT_LIST_DIR}/modules)
list(APPEND CMAKE_MODULE_PATH ${ASCEND_CMAKE_PATH})

if(NOT DEFINED BISHENG_TIMESTAMP)
    find_program(BISHENG_COMPILER NAMES bisheng HINTS ${ASCEND_HOME_PATH}/compiler/ccec_compiler/bin)

    execute_process(
        COMMAND bash -c "${BISHENG_COMPILER} --version 2>&1 | awk 'NR==1{print $1}'"
        OUTPUT_VARIABLE BISHENG_VERSION_STR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE
    )

    string(REGEX REPLACE "[-T:]" "" BISHENG_VERSION_STR ${BISHENG_VERSION_STR})
    string(REGEX REPLACE "^([^+]+).*" "\\1" BISHENG_VERSION_STR ${BISHENG_VERSION_STR})
    set(BISHENG_TIMESTAMP "${BISHENG_VERSION_STR}")
    message("BISHENG_TIMESTAMP: ${BISHENG_VERSION_STR}")
    add_compile_definitions(BISHENG_TIMESTAMP=${BISHENG_TIMESTAMP})
endif()

set(CATLASS_FOUND TRUE)
set(CATLASS_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR}/../)

find_path(CATLASS_INCLUDE_DIR NAMES catlass/catlass.hpp
    HINTS ${CATLASS_ROOT_DIR}/include
)
mark_as_advanced(CATLASS_INCLUDE_DIR)

set(CATLASS_INCLUDE_DIRS ${CATLASS_INCLUDE_DIR})