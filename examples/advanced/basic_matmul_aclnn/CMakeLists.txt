# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set(MSOPGEN_PROJECT_SOURCE_DIR ${CMAKE_BINARY_DIR}/basic_matmul_aclnn)
set(MSOPGEN_CMAKE_BINARY_DIR ${MSOPGEN_PROJECT_SOURCE_DIR}/build_out)

if(NOT EXISTS ${MSOPGEN_PROJECT_SOURCE_DIR})
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/tmp)
    execute_process(
        COMMAND chmod 755 ${CMAKE_BINARY_DIR}/tmp
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/CMakeLists.txt
        ${CMAKE_BINARY_DIR}/tmp/op_kernel/CMakeLists.txt
    )
    file(COPY_FILE ${CMAKE_CURRENT_SOURCE_DIR}/catlass_basic_matmul.json ${CMAKE_BINARY_DIR}/tmp/catlass_basic_matmul.json)
    find_program(MSOPGEN NAMES msopgen
        HINTS ${ASCEND_HOME_PATH})
    add_custom_target(generated_msopgen_project
        COMMAND ${CMAKE_COMMAND} -E make_directory ${MSOPGEN_PROJECT_SOURCE_DIR}
        COMMAND ${MSOPGEN} gen -i ${CMAKE_BINARY_DIR}/tmp/catlass_basic_matmul.json -c ai_core-${NPU_MODEL} -lan cpp -out ${MSOPGEN_PROJECT_SOURCE_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/op_host ${MSOPGEN_PROJECT_SOURCE_DIR}/op_host
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel ${MSOPGEN_PROJECT_SOURCE_DIR}/op_kernel
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/tmp/op_kernel/CMakeLists.txt ${MSOPGEN_PROJECT_SOURCE_DIR}/op_kernel
        BYPRODUCTS ${MSOPGEN_PROJECT_SOURCE_DIR}
    )
else()
    add_custom_target(generated_msopgen_project
        COMMENT "msopgen project directory already exists"
    )
endif()

# if(BUILD_MSOPGEN_RUN_PACKAGE)
add_custom_target(basic_matmul_aclnn_run_package
    COMMAND ${MSOPGEN_PROJECT_SOURCE_DIR}/build.sh
    WORKING_DIRECTORY ${MSOPGEN_PROJECT_SOURCE_DIR}
    BYPRODUCTS ${MSOPGEN_CMAKE_BINARY_DIR}/custom_opp_ubuntu.run
)
add_dependencies(basic_matmul_aclnn_run_package generated_msopgen_project)
# endif()

add_executable(basic_matmul_aclnn basic_matmul_aclnn.cpp)
add_dependencies(basic_matmul_aclnn basic_matmul_aclnn_run_package)
target_include_directories(basic_matmul_aclnn PRIVATE
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/include/aclnn
    ${ASCEND_HOME_PATH}/include/experiment/runtime
    ${ASCEND_HOME_PATH}/include/experiment/msprof
    ${MSOPGEN_CMAKE_BINARY_DIR}/autogen)
target_link_directories(basic_matmul_aclnn PRIVATE
    ${MSOPGEN_CMAKE_BINARY_DIR}/op_host)
target_link_libraries(basic_matmul_aclnn PRIVATE cust_opapi)

execute_process(COMMAND grep -i ^id= /etc/os-release OUTPUT_VARIABLE TEMP)
string(REGEX REPLACE "\n|id=|ID=|\"" "" SYSTEM_NAME ${TEMP})
set(SYSTEM_INFO "${SYSTEM_NAME}_${CMAKE_SYSTEM_PROCESSOR}")
set(PACKAGE_NAME "custom_opp_${SYSTEM_INFO}.run")

install(
    FILES "${MSOPGEN_CMAKE_BINARY_DIR}/${PACKAGE_NAME}"
    DESTINATION run
    COMPONENT basic_matmul_aclnn
)

install(TARGETS basic_matmul_aclnn
    RUNTIME DESTINATION bin/advanced
    COMPONENT basic_matmul_aclnn
)
