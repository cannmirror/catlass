import os
import re
import subprocess
from typing import Dict

from inflection import underscore
import logging
from catlass_test import (
    CATLASS_TEST_KERNEL_PATH,
)

from catlass_test.compiler.options import (
    COMPILER_COMPILE_OPTIONS,
    COMPILER_INCLUDE_DIRECTORIES,
    COMPILER_LINK_DIRECTORIES,
    COMPILER_LINK_LIBRARIES,
    COMPILER_LLVM_COMPILE_OPTIONS,
    CATLASS_KERNEL_ENTRY_FILE,
)
from catlass_test.common import OpType


def list_to_dict(s):
    return {
        parts[1]: parts[0]
        for item in [x.strip() for x in s.split(",")]
        for parts in [item.split()]
        if len(parts) == 2
    }


class TemplateCompiler:
    def __init__(self, kernel_template_src: str):
        self.kernel_template_src = kernel_template_src
        self.__init_kernel_name_and_params()

    def __init_kernel_name_and_params(self):
        pattern = re.compile(
            r"template\s*<([^>]+)>\s*(?:inline\s+)?TEMPLATE_RET_TYPE\s+(\w+)\s*\(([^)]+)\)"
        )
        with open(self.kernel_template_src, mode="r+") as kernel_template_src_handle:
            match = pattern.search(kernel_template_src_handle.read())
            if match is not None:
                self.compile_params = list_to_dict(match.group(1))
                self.kernel_name = match.group(2)
                self.orig_runtime_params = match.group(3)
                runtime_params = list_to_dict(match.group(3))
                self.runtime_params = {}
                for var_name, var_type in runtime_params.items():
                    if var_name.startswith("*") or var_name.startswith("&"):
                        self.runtime_params[var_name[1:]] = f"{var_type}*"
                    else:
                        self.runtime_params[var_name] = var_type

    @property
    def runtime_params_call(self):
        return ",".join(self.runtime_params.keys())

    def compile(
        self, compile_definitions: Dict[str, str], op_type: OpType = OpType.MIX_AIC_1_2
    ) -> str:
        """编译算子"""

        dcompile_params = []
        for var_name in self.compile_params.keys():
            dcompile_params.append(compile_definitions.get(var_name))
        compile_params = ",".join(dcompile_params)
        kernel_name = (
            "_".join(
                [f"lib{underscore(self.kernel_name)}"]
                + [
                    dcompile_param.split("::")[-1].lower()
                    for dcompile_param in dcompile_params
                ]
            )
            + ".so"
        )
        logging.info(f"compiling kernel {kernel_name}")
        kernel_full_path = os.path.join(CATLASS_TEST_KERNEL_PATH, kernel_name)
        if os.path.exists(kernel_full_path):
            return kernel_full_path
        COMPILER_DEFINATIONS = [
            f"-DKERNEL_TEMPLATE_FILE={self.kernel_template_src}",
            f"-DCOMPILE_PARAM={compile_params}",
            f"-DRUNTIME_PARAM={self.orig_runtime_params}",
            f"-DRUNTIME_PARAM_CALL={self.runtime_params_call}",
            f"-DKERNEL_TEMPLATE_NAME={self.kernel_name}",
            f"-DTILING_KEY_VAR",
            f"-D{op_type.name}",
        ]
        command = (
            ["bisheng"]
            + COMPILER_COMPILE_OPTIONS
            + COMPILER_DEFINATIONS
            + COMPILER_LLVM_COMPILE_OPTIONS
            + COMPILER_INCLUDE_DIRECTORIES
            + COMPILER_LINK_DIRECTORIES
            + COMPILER_LINK_LIBRARIES
            + [
                "-fPIC",
                "--shared",
                CATLASS_KERNEL_ENTRY_FILE,
                "-o",
                kernel_full_path,
            ]
        )

        # 执行命令
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logging.error(
                "Compile failed! The return code of compiler is not 0. Error info: "
            )
            logging.error(result.stderr)
            logging.error("Compile command:")
            logging.error(" ".join(command))
            exit(1)
        return kernel_full_path
