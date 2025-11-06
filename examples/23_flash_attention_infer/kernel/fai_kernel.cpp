/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#include "fai.h"

template <class DataType>
CATLASS_GLOBAL void FAInfer(
    uint64_t fftsAddr,
    GM_ADDR q,
    GM_ADDR k,
    GM_ADDR v,
    GM_ADDR mask,
    GM_ADDR blockTables,
    GM_ADDR o,
    GM_ADDR actualQseqlen,
    GM_ADDR actualKvseqlen,
    GM_ADDR s,
    GM_ADDR p,
    GM_ADDR oTemp,
    GM_ADDR oUpdate,
    GM_ADDR tiling
)
{
    AscendC::SetSyncBaseAddr(fftsAddr);

    using ArchTag = Arch::AtlasA2;
    using ElementQ = DataType;
    using LayoutQ = layout::RowMajor;
    using ElementK = DataType;
    using LayoutK = layout::ColumnMajor;
    using ElementV = DataType;
    using LayoutV = layout::RowMajor;
    using ElementS = float;
    using LayoutS = layout::RowMajor;
    using ElementP = DataType;
    using LayoutP = layout::RowMajor;
    using ElementO = DataType;
    using LayoutO = layout::RowMajor;
    using ElementMask = DataType;
    using LayoutMask = layout::RowMajor;
    using ElementOTmp = float;
    using LayoutOTmp = layout::RowMajor;
    using ElementUpdate = float;
    using LayoutUpdate = layout::RowMajor;
    // L1TileShape::K must be embdding
    using L1TileShape = GemmShape<128, 128, 128>;
    using L0TileShape = L1TileShape;
    // GEMM Block模块，实现Flash Attention Infer的Q * K^T
    // using DispatchPolicyQK = Gemm::MmadAtlasA2FAIQK<true>;
    using DispatchPolicyQK = Gemm::MmadAtlasA2FAIQK<true, false>;
    using QType = Gemm::GemmType<ElementQ, LayoutQ>;
    using KType = Gemm::GemmType<ElementK, LayoutK>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using BlockMmadQK = Gemm::Block::BlockMmad<DispatchPolicyQK, L1TileShape, L0TileShape, QType, KType, SType>;

    using DispatchPolicyQKTail = Gemm::MmadAtlasA2FAITailQK<true, false>;
    using QType = Gemm::GemmType<ElementQ, LayoutQ>;
    using KType = Gemm::GemmType<ElementK, LayoutK>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using BlockMmadQKTail = Gemm::Block::BlockMmad<DispatchPolicyQKTail, L1TileShape, L0TileShape, QType, KType, SType>;

    // Epilogue Block模块，实现Flash Attention Infer中当前S基块的softmax
    using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueAtlasA2OnlineSoftmax;
    using PType = Gemm::GemmType<ElementP, LayoutP>;
    using maskType = Gemm::GemmType<ElementMask, LayoutMask>;
    using EpilogueOnlineSoftmax = Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType, maskType>;

    // GEMM Block模块，实现Flash Attention Infer的P * V
    // using DispatchPolicyPV = Gemm::MmadAtlasA2FAIPV<true>;
    using DispatchPolicyPV = Gemm::MmadAtlasA2FAIPV<true, false>;
    using VType = Gemm::GemmType<ElementV, LayoutV>;
    using OTmpType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
    using BlockMmadPV = Gemm::Block::BlockMmad<DispatchPolicyPV, L1TileShape, L0TileShape, PType, VType, OTmpType>;

    using DispatchPolicyPVTail = Gemm::MmadAtlasA2FAITailPV<true, false>;
    using VType = Gemm::GemmType<ElementV, LayoutV>;
    using OTmpType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
    using BlockMmadPVTail =
        Gemm::Block::BlockMmad<DispatchPolicyPVTail, L1TileShape, L0TileShape, PType, VType, OTmpType>;

    // Epilogue Block模块，实现Flash Attention Infer中当前O基块的更新
    using DispatchPolicyRescaleO = Epilogue::EpilogueAtlasA2RescaleO;
    using OType = Gemm::GemmType<ElementO, LayoutO>;
    using OUpdateType = Gemm::GemmType<ElementUpdate, LayoutUpdate>;
    using EpilogueRescaleO = Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, OType, OTmpType, OUpdateType>;

    // Kernel level
    // using FAInferKernel = FAInferKernel<BlockMmadQK, BlockMmadPV, EpilogueOnlineSoftmax, EpilogueRescaleO, true>;
    using FAInferKernel = FAInferKernel<
        BlockMmadQK, BlockMmadPV, BlockMmadQKTail, BlockMmadPVTail, EpilogueOnlineSoftmax, EpilogueRescaleO, true>;
    FAIKernelParams params{q, k, v, mask, blockTables, actualQseqlen, actualKvseqlen, o, s, p, oTemp, oUpdate, tiling};

    // call kernel
    FAInferKernel flashAttnInfer;
    flashAttnInfer(params);
}

void FAInferDevice(
    uint8_t blockNum,
    aclrtStream stream,
    aclDataType dataType,
    uint64_t fftsAddr,
    uint8_t *q,
    uint8_t *k,
    uint8_t *v,
    uint8_t *mask,
    uint8_t *blockTables,
    uint8_t *o,
    uint8_t *actualQseqlen,
    uint8_t *actualKvseqlen,
    uint8_t *s,
    uint8_t *p,
    uint8_t *oTemp,
    uint8_t *oUpdate,
    uint8_t *tiling
)
{
    if (dataType == ACL_FLOAT16) {
        FAInfer<half><<<blockNum, nullptr, stream>>>(
            q, k, v, mask, blockTables, actualQseqlen, actualKvseqlen, o, s, p, oTemp, oUpdate, tiling
        );
    } else if (dataType == ACL_BF16) {
        FAInfer<bfloat16_t><<<blockNum, nullptr, stream>>>(
            q, k, v, mask, blockTables, actualQseqlen, actualKvseqlen, o, s, p, oTemp, oUpdate, tiling
        );
    }
}