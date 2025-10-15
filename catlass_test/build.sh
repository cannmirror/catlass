#!/bin/bash
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================

CATLASS_REPO_URL=https://gitcode.com/cann/catlass.git
CATLASS_TAG=$1

git clone -b $CATLASS_TAG --depth=1 $CATLASS_REPO_URL /tmp/catlass_test_build
CATLASS_COMMIT_ID=$(git -C /tmp/catlass_test_build/ rev-parse HEAD)
CATLASS_COMMIT_ID_SHORT=$(git -C /tmp/catlass_test_build/ rev-parse --short HEAD) 

mv /tmp/catlass_test_build/include/catlass catlass_test/csrc/include/
mv /tmp/catlass_test_build/include/tla catlass_test/csrc/include/

sed -i "s/CATLASS_COMMIT_ID=\"\"/CATLASS_COMMIT_ID=\"${CATLASS_COMMIT_ID}\"/" catlass_test/__init__.py
sed -i "s/\(version = \"[0-9]\+\.[0-9]\+\.[0-9]\+\)\"/\1+${CATLASS_COMMIT_ID_SHORT}\"/" pyproject.toml
uv build 