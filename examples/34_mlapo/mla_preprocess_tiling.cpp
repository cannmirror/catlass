/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

struct MlaPreprocessTilingData {
    // basic
    uint32_t rmsNumCol2 = 0;
    uint32_t n = 0;
};

namespace MlaPreprocessTiling {

struct MlaPreprocessInfo {
    uint32_t rmsNumCol2 = 0;
    uint32_t n = 0;
};

void FillBasicTilingData(const MlaPreprocessInfo &mpInfo, MlaPreprocessTilingData &mpTilingData) {
    mpTilingData.rmsNumCol2 = mpInfo.rmsNumCol2;
    mpTilingData.n = mpInfo.n;
}

int32_t GetMpTilingParam(const MlaPreprocessInfo &mpInfo, MlaPreprocessTilingData &mpTilingData) {
    FillBasicTilingData(mpInfo, mpTilingData);
    return 0;
}
} // namespace MlaPreprocessTiling