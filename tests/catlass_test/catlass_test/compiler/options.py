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
