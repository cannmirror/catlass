/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef KERNEL_COMMON
#define KERNEL_COMMON

constexpr uint32_t QK_READY_ID = 1;
constexpr uint32_t SOFTMAX_READY_ID = 2;
constexpr uint32_t PV_READY_ID = 3;
constexpr uint32_t BLOCK_SIZE = 16;
constexpr uint32_t PRE_LAUNCH = 2;
constexpr uint32_t MAX_STACK_LEN = 512;
constexpr uint32_t Q_BLK = 128;
constexpr uint32_t WORKSPACE_BLOCK_SIZE_DB = Q_BLK * MAX_STACK_LEN;

CATLASS_HOST_DEVICE
uint32_t GetQNBlockTile(uint32_t qSeqlen, uint32_t groupSize)
{
    uint32_t qNBlockTile = (Q_BLK / qSeqlen) / 2 * 2;
    qNBlockTile = qNBlockTile < groupSize ? qNBlockTile : groupSize;
    qNBlockTile = qNBlockTile < 1 ? 1 : qNBlockTile;
    return qNBlockTile;
}

CATLASS_HOST_DEVICE
uint32_t GetQSBlockTile(uint32_t kvSeqlen)
{
    uint32_t qSBlockTile = Q_BLK;
    return qSBlockTile;
}

struct FATilingData {
    uint32_t numHeads = 0;
    uint32_t embeddingSize = 0;
    uint32_t numBlocks = 0;
    uint32_t blockSize = 0;
    uint32_t maxKvSeqlen = 0;
    uint32_t kvHeads = 0;
    uint32_t batch = 0;
    uint32_t maxNumBlocksPerBatch = 0;
    uint32_t firstBatchTaskNum = 0;
    uint32_t totalTaskNum = 0;
    uint32_t maskType = 0;
    uint64_t mm1OutSize = 0;
    uint64_t smOnlineOutSize = 0;
    uint64_t mm2OutSize = 0;
    uint64_t UpdateSize = 0;
    uint64_t workSpaceSize = 0;
    float scaleValue = 0.0;
};

struct FAIKernelParams {
    GM_ADDR q;
    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR mask;
    GM_ADDR blockTables;
    GM_ADDR actualQseqlen;
    GM_ADDR actualKvseqlen;
    GM_ADDR o;
    GM_ADDR s;
    GM_ADDR p;
    GM_ADDR oTemp;
    GM_ADDR oUpdate;
    GM_ADDR tiling;
    // Methods
    CATLASS_DEVICE
    FAIKernelParams()
    {}
    CATLASS_DEVICE
    FAIKernelParams(GM_ADDR q_, GM_ADDR k_, GM_ADDR v_, GM_ADDR mask_, GM_ADDR blockTables_, GM_ADDR actualQseqlen_,
        GM_ADDR actualKvseqlen_, GM_ADDR o_, GM_ADDR s_, GM_ADDR p_, GM_ADDR oTemp_, GM_ADDR oUpdate_, GM_ADDR tiling_)
        : q(q_), k(k_), v(v_), mask(mask_), blockTables(blockTables_), actualQseqlen(actualQseqlen_),
          actualKvseqlen(actualKvseqlen_), o(o_), s(s_), p(p_), oTemp(oTemp_), oUpdate(oUpdate_), tiling(tiling_)
    {}
};

#endif