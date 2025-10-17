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

import logging
import os
import shutil
from importlib.metadata import PackageNotFoundError, version

ASCEND_HOME_PATH = os.environ["ASCEND_HOME_PATH"]

CATLASS_TEST_PATH = os.path.dirname(__file__)
CATLASS_TEST_TMP_PATH = os.path.join("/tmp", "catlass_test")
CATLASS_TEST_KERNEL_PATH = os.path.join(CATLASS_TEST_TMP_PATH, "kernel")
CATLASS_TEST_INCLUDE_PATH = os.path.join(CATLASS_TEST_PATH, "csrc", "include")

CATLASS_PATH = os.path.join(CATLASS_TEST_TMP_PATH, "catlass")
CATLASS_INCLUDE_PATH = CATLASS_TEST_INCLUDE_PATH

CATLASS_COMMIT_ID = ""

try:
    CATLASS_COMMIT_ID = version("catlass_test").split("+")[-1]
except PackageNotFoundError:
    pass

__LOG_LEVEL = logging._nameToLevel.get(
    os.environ.get("CATLASS_TEST_LOG_LEVEL", "INFO").upper(), logging.INFO
)

logging.basicConfig(
    level=__LOG_LEVEL,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

os.makedirs(CATLASS_TEST_TMP_PATH, exist_ok=True)

if os.environ.get("CATLASS_TEST_CLEAN_KERNEL_CACHE"):
    shutil.rmtree(CATLASS_TEST_KERNEL_PATH, ignore_errors=True)
os.makedirs(CATLASS_TEST_KERNEL_PATH, exist_ok=True)

from catlass_test.interface.function import *
