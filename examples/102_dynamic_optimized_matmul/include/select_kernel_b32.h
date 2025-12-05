/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SELECT_KERNEL_FLOAT_H
#define SELECT_KERNEL_FLOAT_H

#include "platform_info.h"
#include <limits>

double GetBandwidthB32(uint32_t nValue, uint32_t dValue, uint32_t srcDValue, double& MAX_BANDWIDTH_AIC) {
    double a6 = -0.000000000000009004592101;
    double a5 = 0.000000000019396510970002;
    double a4 = -0.00000001611519890867614;
    double a3 = 0.000006657887925107158;
    double a2 = -0.0015784588599109292;
    double a1 = 0.3035859339962211;
    double a0 = 4.295633544322609;
    double unalignBand = a6 * pow(static_cast<double>(dValue), 6) + a5 * pow(static_cast<double>(dValue), 5)
    + a4 * pow(static_cast<double>(dValue), 4) + a3 * pow(static_cast<double>(dValue), 3)
    + a2 * pow(static_cast<double>(dValue), 2) + a1 * static_cast<double>(dValue) + a0;

    if (dValue == srcDValue) {
        if (dValue <= 64 && dValue>=16 && dValue % 8 == 0) {
            unalignBand = 95;
        }
        else if (dValue % 128 == 0) {
            unalignBand = 95;
        }
        else if (dValue % 64 == 0) {
            unalignBand += 40;
        }
        else if (dValue % 32 == 0) {
            unalignBand += 17;
        }
    }

    if (srcDValue >= 65536) {
        unalignBand = 1;
    }

    if (srcDValue % 128 == 0) {
        unalignBand = static_cast<double>(100) / 40 * unalignBand;
    } else if (srcDValue % 64 == 0) {
        unalignBand = static_cast<double>(80) / 40 * unalignBand;
    } else if (srcDValue % 32 == 0) {
        unalignBand = static_cast<double>(60) / 40 * unalignBand;
    } else if (srcDValue % 16 == 0) {
        unalignBand = static_cast<double>(50) / 40 * unalignBand;
    }

    unalignBand = std::min(unalignBand, MAX_BANDWIDTH_AIC);

    if (dValue % 256 == 0) {
        if (nValue < 16) {
            double b2 = -0.004727824537240172;
            double b1 = 0.13718894235979967;
            double b0 = 0.015569563581479779;
            unalignBand = unalignBand * (b2 * pow(nValue, 2) + b1 * pow(nValue, 1) + b0);
        }
    } else if (dValue % 32 == 0) {
        if (nValue < 32) {
            double b2 = -0.0008410977863908918;
            double b1 = 0.055534598197727145;
            double b0 = 0.009854896921994448;
            unalignBand = unalignBand * (b2 * pow(nValue, 2) + b1 * pow(nValue, 1) + b0);
        }
    } else {
        if (nValue < 64) {
            double b3 = 0.0000011791112126816647;
            double b2 = -0.0004494844311314906;
            double b1 = 0.03868377611449111;
            double b0 = 0.014540735965952572;
            unalignBand = unalignBand * (b3 * pow(nValue, 3) + b2 * pow(nValue, 2) + b1 * pow(nValue, 1) + b0);
        }
    }
    return unalignBand;
}

void GetPaddingTagB32(TilingParams& tilingParams, PlatformInfo& platformInfo) {
    uint32_t m = tilingParams.m;
    uint32_t n = tilingParams.n;
    uint32_t k = tilingParams.k;
    uint32_t m1 = tilingParams.m1;
    uint32_t n1 = tilingParams.n1;
    uint32_t k1 = tilingParams.k1;
    uint32_t splitkFactor = tilingParams.splitkFactor;

    double MAX_BANDWIDTH_AIC = 1.6 * 1024 / platformInfo.coreNum;

    uint64_t outterAxisA = m;
    uint64_t innerAxisA = k;
    uint32_t nValueA = std::min(m, m1);
    uint32_t dValueA = std::min(k, k1);

    uint64_t outterAxisB = k;
    uint64_t innerAxisB = n;
    uint32_t nValueB = std::min(k, k1);
    uint32_t dValueB = std::min(n, n1);

    if (static_cast<LayoutTag>(tilingParams.layoutTagA) == LayoutTag::TagColumnMajor) {
        outterAxisA = k;
        innerAxisA = m;
        nValueA = std::min(k, k1);
        dValueA = std::min(m, m1);
    }
    if (static_cast<LayoutTag>(tilingParams.layoutTagB) == LayoutTag::TagColumnMajor) {
        outterAxisB = n;
        innerAxisB = k;
        nValueB = std::min(n, n1);
        dValueB = std::min(k, k1);
    }

    double aBandwidthAiv = 30; // single core GB/s
    uint64_t matrixASize = static_cast<size_t>(m) * k * 4; // float type
    if (matrixASize > 192 * 1024 * 1024) { // L2 Cache size
        aBandwidthAiv = 10;
    }
    double aBandwidthBeforePaddingAic = GetBandwidthB32(nValueA, dValueA, innerAxisA, MAX_BANDWIDTH_AIC);

    uint32_t tasksAic = CeilDiv(m, m1) * CeilDiv(n, n1);
    uint32_t blockDimAic = tasksAic > platformInfo.coreNum ? platformInfo.coreNum : tasksAic;
    if (CeilDiv(m, m1) < blockDimAic / 2 && k <= k1 && CeilDiv(m, m1) <= 2) {
        aBandwidthBeforePaddingAic = aBandwidthBeforePaddingAic / (blockDimAic / CeilDiv(m, m1)) * 1.5;
    }
    double aBandwidthAfterPaddingAic = MAX_BANDWIDTH_AIC;

    double bBandwidthAiv = 30; // single core GB/s
    uint64_t matrixBSize = static_cast<size_t>(k) * n * 4;
    if (matrixBSize > 192 * 1024 * 1024) {
        bBandwidthAiv = 10;
    }
    double bBandwidthBeforePaddingAic = GetBandwidthB32(nValueB, dValueB, innerAxisB, MAX_BANDWIDTH_AIC);
    if (CeilDiv(n, n1) < blockDimAic / 2 && k <= k1 && CeilDiv(n, n1) <= 2) {
        bBandwidthBeforePaddingAic = bBandwidthBeforePaddingAic / (blockDimAic / CeilDiv(n, n1)) * 1.5;
    }
    double bBandwidthAfterPaddingAic = MAX_BANDWIDTH_AIC; // single core GB/s

    uint32_t actualM = std::min(m, m1);
    uint32_t actualN = std::min(n, n1);
    uint32_t roundMax = CeilDiv(CeilDiv(m, m1) * CeilDiv(n, n1) * splitkFactor, platformInfo.coreNum);
    uint64_t aMaxDataSizeAic = static_cast<size_t>(roundMax) * actualM * CeilDiv(k, splitkFactor) * 4; // Byte
    uint64_t bMaxDataSizeAic = static_cast<size_t>(roundMax) * actualN * CeilDiv(k, splitkFactor) * 4; // Byte

    // padding simulator
    uint64_t aMaxDataSizeAiv{0};
    uint32_t tasksAivA{0};
    {
        uint32_t taskRows = 16;
        uint32_t taskCols = 48 * 1024 / 4 / taskRows;
        if (innerAxisA < taskCols) {
            taskCols = innerAxisA;
        }
        if (outterAxisA < taskRows) {
            taskRows = outterAxisA;
        }
        taskCols = RoundUp(innerAxisA / CeilDiv(innerAxisA, taskCols), 8);
        uint32_t tasksAivA = CeilDiv(outterAxisA, taskRows) * CeilDiv(innerAxisA, taskCols);
        uint32_t maxTasksPerCore = CeilDiv(tasksAivA, platformInfo.coreNum * 2);
        aMaxDataSizeAiv = maxTasksPerCore * taskCols * taskRows * 4;
    }
    uint64_t bMaxDataSizeAiv{0};
    uint32_t tasksAivB{0};
    {
        uint32_t taskRows = 16;
        // 48KB is UB buffer size
        uint32_t taskCols = 48 * 1024 / 4 / taskRows;
        if (innerAxisB < taskCols) {
            taskCols = innerAxisB;
        }
        if (outterAxisB < taskRows) {
            taskRows = outterAxisB;
        }
        taskCols = RoundUp(innerAxisB / CeilDiv(innerAxisB, taskCols), 8);
        uint32_t tasksAivB = CeilDiv(outterAxisB, taskRows) * CeilDiv(innerAxisB, taskCols);
        uint32_t maxTasksPerCore = CeilDiv(tasksAivB, platformInfo.coreNum * 2);
        bMaxDataSizeAiv = maxTasksPerCore * taskCols * taskRows * 4;
    }
    tasksAic = CeilDiv(m, m1) * CeilDiv(n, n1) * tilingParams.splitkFactor;
    uint32_t tasksAiv = std::max(tasksAivA, tasksAivB);

    uint32_t blockDimAiv = CeilDiv(tasksAiv, 2) > platformInfo.coreNum ? platformInfo.coreNum : CeilDiv(tasksAiv, 2);
    if (innerAxisA > 192 && innerAxisB > 192) {
        blockDimAic = std::max(blockDimAic, blockDimAiv);
    }

    double headCost = 1 + 7 * static_cast<double>(blockDimAic) / platformInfo.coreNum; // us
    double t00 = static_cast<double>(aMaxDataSizeAic) / aBandwidthBeforePaddingAic / 1000
        + static_cast<double>(bMaxDataSizeAic) / bBandwidthBeforePaddingAic / 1000;
    double t01 = static_cast<double>(aMaxDataSizeAic) / aBandwidthBeforePaddingAic / 1000
        + static_cast<double>(bMaxDataSizeAic) / bBandwidthAfterPaddingAic / 1000
        + static_cast<double>(bMaxDataSizeAiv) / bBandwidthAiv / 1000 + headCost;
    double t10 = static_cast<double>(aMaxDataSizeAic) / aBandwidthAfterPaddingAic / 1000
        + static_cast<double>(bMaxDataSizeAic) / bBandwidthBeforePaddingAic / 1000
        + static_cast<double>(aMaxDataSizeAiv) / aBandwidthAiv / 1000 + headCost;
    double t11 = static_cast<double>(aMaxDataSizeAic) / aBandwidthAfterPaddingAic / 1000
        + static_cast<double>(bMaxDataSizeAic) / bBandwidthAfterPaddingAic / 1000
        + static_cast<double>(aMaxDataSizeAiv) / aBandwidthAiv / 1000
        + static_cast<double>(bMaxDataSizeAiv) / bBandwidthAiv / 1000 + headCost + 2;

    double minCost = std::numeric_limits<double>::max();
    PaddingTag paddingTagA = PaddingTag::PADDING_NONE;
    PaddingTag paddingTagB = PaddingTag::PADDING_NONE;
    if (minCost > t00) {
        minCost = t00;
    }
    if (minCost > t01) {
        minCost = t01;
        paddingTagA = PaddingTag::PADDING_NONE;
        paddingTagB = PaddingTag::PADDING_NZ;
    }
    if (minCost > t10) {
        minCost = t10;
        paddingTagA = PaddingTag::PADDING_NZ;
        paddingTagB = PaddingTag::PADDING_NONE;
    }
    if (minCost > t11) {
        minCost = t11;
        paddingTagA = PaddingTag::PADDING_NZ;
        paddingTagB = PaddingTag::PADDING_NZ;
    }

    if ((innerAxisA < 8 || (innerAxisA < 32 && (innerAxisA % 16 != 0))) && outterAxisA > 512) {
        paddingTagA = PaddingTag::PADDING_NZ;
    }
    if ((innerAxisB < 8 || (innerAxisB < 32 && (innerAxisB % 16 != 0))) && outterAxisB > 512) {
        paddingTagB = PaddingTag::PADDING_NZ;
    }

    PaddingTag paddingTagC = PaddingTag::PADDING_NONE;
    if (static_cast<size_t>(m) * n > 2048 * 2048 && n > 256 && (n % 128 != 0)) {
        size_t totalDataSize = static_cast<size_t>(m) * k * CeilDiv(n, n1) * 4
            + static_cast<size_t>(k) * n * CeilDiv(m, m1) * 4 + static_cast<size_t>(m) * n * 4;
        if (totalDataSize < 192 * 1024 * 1024) { // L2 cache size
            paddingTagC = PaddingTag::PADDING_ND;
        }
    }

    tilingParams.paddingTagA = static_cast<uint8_t>(paddingTagA);
    tilingParams.paddingTagB = static_cast<uint8_t>(paddingTagB);
    tilingParams.paddingTagC = static_cast<uint8_t>(paddingTagC);

    uint32_t blockDim = blockDimAic;
    uint32_t actualTasksAivA{0};
    uint32_t actualTasksAivB{0};
    if (tilingParams.paddingTagA && innerAxisA > 192) {
        actualTasksAivA = tasksAivA;
    }
    if (tilingParams.paddingTagB && innerAxisB > 192) {
        actualTasksAivB = tasksAivB;
    }
    uint32_t actualTasksAiv = std::max(actualTasksAivA, actualTasksAivB);
    blockDimAiv = CeilDiv(actualTasksAiv, 2) > platformInfo.coreNum
        ? platformInfo.coreNum : CeilDiv(actualTasksAiv, 2);
    if (tilingParams.paddingTagA || tilingParams.paddingTagB) {
        blockDim = std::max(blockDimAic, blockDimAiv);
    }
    tilingParams.blockDim = blockDim;
}

bool SmallMatmulB32Handler(TilingParams& params, PlatformInfo& platformInfo) {
    uint8_t kernelSerial = 1;
    GetPaddingTagB32(params, platformInfo);

    if (static_cast<PaddingTag>(params.paddingTagA) == PaddingTag::PADDING_NONE
        && static_cast<PaddingTag>(params.paddingTagB) == PaddingTag::PADDING_NONE
        && static_cast<PaddingTag>(params.paddingTagC) == PaddingTag::PADDING_NONE) {

        uint32_t taskBlocks = CeilDiv(params.m, params.m1) * CeilDiv(params.n, params.n1);
        if (taskBlocks <= platformInfo.coreNum && params.k <= params.k1) {
            params.tilingKey.SetTilingKey(kernelSerial, params.layoutTagA, params.layoutTagB, 0, 0, 0, 0, 1);
            return true;
        }
    }
    return false;
}

bool PaddingCommonMatmulB32Handler(TilingParams& params, PlatformInfo& platformInfo) {
    uint8_t kernelSerial = 2;
    if (params.paddingTagA || params.paddingTagB || params.paddingTagC) {
        params.tilingKey.SetTilingKey(kernelSerial,
            params.layoutTagA, params.layoutTagB, 0, params.paddingTagA, params.paddingTagB, params.paddingTagC, 1);
        return true;
    }
    return false;
}

bool CommonMatmulB32Handler(TilingParams& params, PlatformInfo& platformInfo) {
    uint8_t kernelSerial = 0;
    uint32_t taskBlocks = CeilDiv(params.m, params.m1) * CeilDiv(params.n, params.n1);
    params.blockDim = taskBlocks > platformInfo.coreNum ? platformInfo.coreNum : taskBlocks;

    // kernelSerial, layoutTagA, layoutTagB, layoutTagC, paddingTagA, paddingTagB, paddingTagC, dtype(defalut 0).
    params.tilingKey.SetTilingKey(kernelSerial, params.layoutTagA, params.layoutTagB, 0, 0, 0, 0, 1);
    return true;
}

bool PaddingMultiCoreSplitkMatmulB32Handler(TilingParams& params, PlatformInfo& platformInfo) {
    if (params.k <= 128) {
        params.splitkFactor = 1;
        return false;
    }
    uint32_t m1 = params.m1;
    uint32_t n1 = params.n1;
    uint32_t k1 = params.k1;
    uint32_t orgBlocks = CeilDiv(params.m, m1) * CeilDiv(params.n, n1);
    uint32_t m1t = 128, n1t = 256, k1t = 128;
    LayoutTag tagA = static_cast<LayoutTag>(params.layoutTagA);
    LayoutTag tagB = static_cast<LayoutTag>(params.layoutTagB);
    bool cond1 = (tagA == LayoutTag::TagColumnMajor && tagB == LayoutTag::TagColumnMajor);
    bool cond2 = ((tagA == LayoutTag::TagColumnMajor && tagB == LayoutTag::TagRowMajor) && (params.m > params.n));
    if (cond1 || cond2) {
        m1t = 256;
        n1t = 128;
    }
    uint32_t blocks = CeilDiv(params.m, m1t) * CeilDiv(params.n, n1t);
    uint32_t maxSplitkFactor = 2;
    if (params.k > 1024) {
        maxSplitkFactor = 4;
    }
    if (params.k > 8192) {
        maxSplitkFactor = 8;
    }
    if (params.k >= 12288) {
        maxSplitkFactor = platformInfo.coreNum;
    }
    params.splitkFactor = std::min(platformInfo.coreNum / blocks, maxSplitkFactor);
    uint32_t newBlocks = blocks * params.splitkFactor;
    // 切核变少了，不使用该kernel
    if (newBlocks < orgBlocks) {
        params.splitkFactor = 1;
        return false;
    }
    if ((blocks <= platformInfo.coreNum / 2 && params.k > 5120) || (blocks <= 2 && params.k > 1024)) {
        params.m1 = m1t;
        params.n1 = n1t;
        params.k1 = k1t;
        uint8_t kernelSerial = 3;
        params.splitkFactor = std::min(platformInfo.coreNum / blocks, maxSplitkFactor);
        GetPaddingTagB32(params, platformInfo);
        params.tilingKey.SetTilingKey(
            kernelSerial, params.layoutTagA, params.layoutTagB, 0, params.paddingTagA, params.paddingTagB, 0, 1);
        return true;
    }
    return false;
}

bool PaddingStreamkMatmulB32Handler(TilingParams& params, PlatformInfo& platformInfo) {
    uint32_t m1 = params.m1;
    uint32_t n1 = params.n1;
    uint32_t k1 = params.k1;
    // Streamk ensures workload balancing by partitioning k, the L1 tile block can use the size with the best bandwidth.
    // The size setting of l1 tile does not need to consider workload balancing.
    uint32_t m1t = 128, n1t = 256, k1t = 128;
    LayoutTag tagA = static_cast<LayoutTag>(params.layoutTagA);
    LayoutTag tagB = static_cast<LayoutTag>(params.layoutTagB);
    bool cond1 = (tagA == LayoutTag::TagColumnMajor && tagB == LayoutTag::TagColumnMajor);
    bool cond2 = ((tagA == LayoutTag::TagColumnMajor && tagB == LayoutTag::TagRowMajor) && (params.m > params.n));
    if (cond1 || cond2) {
        m1t = 256;
        n1t = 128;
    }
    uint32_t blocks = CeilDiv(params.m, m1t) * CeilDiv(params.n, n1t);
    uint32_t skBlocks = blocks % platformInfo.coreNum;
    if (blocks > platformInfo.coreNum && blocks < 8 * platformInfo.coreNum && skBlocks < 0.8 * platformInfo.coreNum
        && params.k > 3072) {
        params.m1 = m1t;
        params.n1 = n1t;
        params.k1 = k1t;
        GetPaddingTagB32(params, platformInfo);
        uint8_t kernelSerial = 4;
        params.tilingKey.SetTilingKey(
            kernelSerial, params.layoutTagA, params.layoutTagB, 0, params.paddingTagA, params.paddingTagB, 0, 1);
        return true;
    }
    return false;
}

void SetSwizzleParamsB32(TilingParams &tilingParams)
{
    if (tilingParams.m > tilingParams.n) {
        tilingParams.swizzleOffset = 3;
        tilingParams.swizzleDirection = 0;
    } else {
        tilingParams.swizzleOffset = 3;
        tilingParams.swizzleDirection = 1;
    }
}

void SelectKernelB32(TilingParams& tilingParams, PlatformInfo& platformInfo)
{
    // Temporarily store the original layoutTagA and layoutTagB
    uint8_t layoutTagATmp = tilingParams.layoutTagA;
    uint8_t layoutTagBTmp = tilingParams.layoutTagB;
    // When m=1 or n=1, the row-major and column-major matrix layouts are identical—the matrix can be stored
    // in either format. In such cases, the layout with higher memory transfer bandwidth should be selected.
    if (tilingParams.m == 1 && static_cast<LayoutTag>(tilingParams.layoutTagA) == LayoutTag::TagColumnMajor) {
        tilingParams.layoutTagA = static_cast<uint8_t>(LayoutTag::TagRowMajor);
    }
    if (tilingParams.n == 1 && static_cast<LayoutTag>(tilingParams.layoutTagB) == LayoutTag::TagRowMajor) {
        tilingParams.layoutTagB = static_cast<uint8_t>(LayoutTag::TagColumnMajor);
    }

    using HandlerPtr = bool (*)(TilingParams& tilingParams, PlatformInfo& platformInfo);
    HandlerPtr Handlers[] = {
        SmallMatmulB32Handler,
        PaddingMultiCoreSplitkMatmulB32Handler,
        PaddingStreamkMatmulB32Handler,
        PaddingCommonMatmulB32Handler,
        CommonMatmulB32Handler
    };

    for (auto handler : Handlers) {
        if (handler(tilingParams, platformInfo)) {
            break;
        }
    }

    // Restore to the original layout
    tilingParams.layoutTagA = layoutTagATmp;
    tilingParams.layoutTagB = layoutTagBTmp;

    SetSwizzleParamsB32(tilingParams);

    uint32_t m = tilingParams.m;
    uint32_t n = tilingParams.n;

    uint32_t m1 = tilingParams.m1;
    uint32_t n1 = tilingParams.n1;
    uint32_t blocks = CeilDiv(m, m1) * CeilDiv(n, n1) * tilingParams.splitkFactor;
    tilingParams.blockDim = blocks > platformInfo.coreNum ? platformInfo.coreNum : blocks;
}

#endif // SELECT_KERNEL_FLOAT_H