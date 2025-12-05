/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ADJUST_TILING_B32_H
#define ADJUST_TILING_B32_H

#include "utils.h"
#include "tiling_params.h"
#include "platform_info.h"

using BlockInfoTuple = std::tuple<uint32_t, uint32_t, uint32_t>;

void DoTilingB32Layout00(TilingParams& tilingParams, PlatformInfo& platformInfo)
{
    uint32_t m = tilingParams.m;
    uint32_t n = tilingParams.n;
    uint32_t k = tilingParams.k;
    uint32_t m1 = 128, n1 = 256, k1 = 128;

    if (n >= 256) {
        // n0 = 256 delivers optimal bandwidth performance.
        uint32_t maxBlocks = RoundUp(CeilDiv(m, m1) * CeilDiv(n, n1), platformInfo.coreNum);
        BalanceWorkload(m, n, m1, n1, 32, platformInfo);
    } else {
        m1 = 256; n1 = 128;
        BalanceWorkload(m, n, m1, n1, 32, platformInfo);
        uint32_t maxBlocks = RoundUp(CeilDiv(m, m1) * CeilDiv(n, n1), platformInfo.coreNum);
        uint32_t m1t = m1;
        while (JudgeSpace<float>(m1t + 16, n1, k1, platformInfo)) {
            m1t += 16;
            uint32_t blocks = CeilDiv(m, m1t) * CeilDiv(n, n1);
            if (blocks <= maxBlocks - platformInfo.coreNum) {
                m1 = m1t;
            }
        }
    }
    if (k >= 65536 || n > 65536) {
        m1 = 128; n1 = 256;
    }
    k1 = GetMaxK1<float>(m1, n1, platformInfo);
    SetTile(tilingParams, m1, n1, k1);
}

void DoTilingB32Layout01(TilingParams& tilingParams, PlatformInfo& platformInfo)
{
    uint32_t m = tilingParams.m;
    uint32_t n = tilingParams.n;
    uint32_t k = tilingParams.k;
    uint32_t m1 = 128, n1 = 256, k1 = 128;
    // When LayoutA is RowMajor and LayoutB is ColumnMajor, bandwidth issues can be completely disregarded,
    // simply choose the tiling configuration with the most balanced workload.
    if (m < n) {
        BalanceWorkload(n, m, n1, m1, 64, platformInfo);
        BalanceWorkload(m, n, m1, n1, 64, platformInfo);
    } else {
        m1 = 256; n1 = 128;
        BalanceWorkload(m, n, m1, n1, 64, platformInfo);
        BalanceWorkload(n, m, n1, m1, 64, platformInfo);
    }
    uint32_t maxBlocks = RoundUp(CeilDiv(m, m1) * CeilDiv(n, n1), platformInfo.coreNum);
    if (m < n) {
        uint32_t n1t = n1;
        while (JudgeSpace<float>(m1, n1t + 16, k1, platformInfo)) {
            n1t += 16;
            uint32_t blocks = CeilDiv(m, m1) * CeilDiv(n, n1t);
            if (blocks <= maxBlocks - platformInfo.coreNum) {
                n1 = n1t;
            }
        }
    } else {
        uint32_t m1t = m1;
        while (JudgeSpace<float>(m1t + 16, n1, k1, platformInfo)) {
            m1t += 16;
            uint32_t blocks = CeilDiv(m, m1t) * CeilDiv(n, n1);
            if (blocks <= maxBlocks - platformInfo.coreNum) {
                m1 = m1t;
            }
        }
    }
    if (k >= 65536) {
        if (m < n) {
            m1 = 128; n1 = 256;
        } else {
            m1 = 256; n1 = 128;
        }
    }
    k1 = GetMaxK1<float>(m1, n1, platformInfo);
    SetTile(tilingParams, m1, n1, k1);
}

void DoTilingB32Layout11(TilingParams& tilingParams, PlatformInfo& platformInfo)
{
    uint32_t m = tilingParams.m;
    uint32_t n = tilingParams.n;
    uint32_t k = tilingParams.k;
    uint32_t m1 = 256, n1 = 128, k1 = 128;
    if (m >= 256) {
        // m0 = 256 delivers optimal bandwidth performance.
        uint32_t maxBlocks = RoundUp(CeilDiv(m, m1) * CeilDiv(n, n1), platformInfo.coreNum);
        BalanceWorkload(n, m, n1, m1, 32, platformInfo);
    } else {
        n1 = 256; m1 = 128;
        BalanceWorkload(n, m, n1, m1, 32, platformInfo);
        uint32_t maxBlocks = RoundUp(CeilDiv(m, m1) * CeilDiv(n, n1), platformInfo.coreNum);
        uint32_t n1t = n1;
        while (JudgeSpace<float>(n1t + 16, m1, k1, platformInfo)) {
            n1t += 16;
            uint32_t blocks = CeilDiv(n, n1t) * CeilDiv(m, m1);
            if (blocks <= maxBlocks - platformInfo.coreNum) {
                n1 = n1t;
            }
        }
    }
    if (k >= 65536 || m > 65536) {
        m1 = 256; n1 = 128;
    }
    k1 = GetMaxK1<float>(m1, n1, platformInfo);
    SetTile(tilingParams, m1, n1, k1);
}

void DoTilingB32Layout10(TilingParams& tilingParams, PlatformInfo& platformInfo)
{
    uint32_t m = tilingParams.m;
    uint32_t n = tilingParams.n;
    uint32_t k = tilingParams.k;

    uint32_t m1 = 128, n1 = 256, k1 = 128;
    if (m > n) {
        m1 = 256; n1 = 128; k1 = 128;
    }
    if (m < m1) {
        m1 = RoundUp(m, 16);
    }
    if (n < n1) {
        n1 = RoundUp(n, 16);
    }

    uint32_t blocks = CeilDiv(m, m1) * CeilDiv(n, n1);
    if (blocks <= platformInfo.coreNum / 4) {
        if (n1 > 16) {
            n1 /= 2;
        }
        if (m1 > 16) {
            m1 /= 2;
        }
    } else if (blocks <= platformInfo.coreNum / 2) {
        if (m1 > n1) {
            m1 /= 2;
        } else if (n1 > 16) {
            n1 /= 2;
        }
    }
    if (n >= 65536 || m > 65536) {
        if (m < n) {
            m1 = 128; n1 = 256;
        } else {
            m1 = 256; n1 = 128;
        }
    }
    k1 = GetMaxK1<float>(m1, n1, platformInfo);
    SetTile(tilingParams, m1, n1, k1);
}


using FuncType = void(*)(TilingParams& tilingParams, PlatformInfo& platformInfo);
std::array<std::array<FuncType, 2>, 2> DoTilingB32 = {{
    {{DoTilingB32Layout00, DoTilingB32Layout01}},
    {{DoTilingB32Layout10, DoTilingB32Layout11}}
}};

#endif // ADJUST_TILING_B32_H