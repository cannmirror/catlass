#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
from setuptools import setup, find_packages
import subprocess
import shutil
import sys
import os

CATLASS_TEST_SETUP_PY_PATH = os.path.dirname(__file__)
CATLASS_REPO_PATH = os.path.realpath(os.path.join(CATLASS_TEST_SETUP_PY_PATH, '..'))
print(CATLASS_REPO_PATH)


def get_version(base_version: str = "0.1.0"):
    """获取 Git 提交 ID"""
    try:
        short_id = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=CATLASS_TEST_SETUP_PY_PATH,
            text=True,
            stderr=subprocess.PIPE,
        ).strip()

        return f"{base_version}+{short_id}"

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error: {e.stderr}") from e
    except FileNotFoundError as e:
        raise RuntimeError(
            "No git executable found. Please install git before building catlass_test."
        ) from e


CATLASS_CATLASS_PATH = os.path.join(
    os.environ.get("CATLASS_REPO_PATH", CATLASS_REPO_PATH), "include", "catlass"
)
CATLASS_TLA_PATH = os.path.join(
    os.environ.get("CATLASS_REPO_PATH", CATLASS_REPO_PATH), "include", "tla"
)


CATLASS_TEST_SETUP_CATLASS_PATH = os.path.join(
    CATLASS_TEST_SETUP_PY_PATH, "catlass_test", "csrc", "include", "catlass"
)
CATLASS_TEST_SETUP_TLA_PATH = os.path.join(
    CATLASS_TEST_SETUP_PY_PATH, "catlass_test", "csrc", "include", "tla"
)

if os.path.exists(CATLASS_TEST_SETUP_CATLASS_PATH):
    shutil.rmtree(CATLASS_TEST_SETUP_CATLASS_PATH)
if os.path.exists(CATLASS_TEST_SETUP_TLA_PATH):
    shutil.rmtree(CATLASS_TEST_SETUP_TLA_PATH)

shutil.copytree(CATLASS_CATLASS_PATH, CATLASS_TEST_SETUP_CATLASS_PATH)
shutil.copytree(CATLASS_TLA_PATH, CATLASS_TEST_SETUP_TLA_PATH)


setup(
    name="catlass_test",
    version=get_version("0.1.0"),
    author="catlass team",
    author_email="catlass@xxx.com",
    description="catlass test framework",
    url="catlass",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "catlass_test": [
            "csrc/include/**/*",  # 包含所有子目录和文件
        ],
    },
)
