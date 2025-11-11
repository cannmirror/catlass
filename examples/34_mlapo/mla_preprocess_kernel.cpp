/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/block/block_RmsNormAndRopeConvergence.hpp"
#include "mla_preprocess_tiling.cpp"
using namespace Catlass;

struct MlaPreprocessKernelParams {
    GM_ADDR gamma3;
    GM_ADDR slotMapping;
    GM_ADDR descale1;
    GM_ADDR s2;
    GM_ADDR s3;
    GM_ADDR sin1;
    GM_ADDR s5;
    GM_ADDR cos1;
    GM_ADDR keycache1;
    GM_ADDR keycache2;
    GM_ADDR quantScale3;
    GM_ADDR tiling;
    // Methods
    CATLASS_DEVICE
    MlaPreprocessKernelParams() {
    }
    CATLASS_DEVICE
    MlaPreprocessKernelParams(GM_ADDR gamma3_,
                    GM_ADDR slotMapping_,
                    GM_ADDR descale1_,
                    GM_ADDR s2_,
                    GM_ADDR s3_,
                    GM_ADDR sin1_,
                    GM_ADDR s5_,
                    GM_ADDR cos1_,
                    GM_ADDR keycache1_,
                    GM_ADDR keycache2_,
                    GM_ADDR quantScale3_,
                    GM_ADDR tiling_)
        : gamma3(gamma3_)
        , slotMapping(slotMapping_)
        , descale1(descale1_)
        , s2(s2_)
        , s3(s3_)
        , sin1(sin1_)
        , s5(s5_)
        , cos1(cos1_)
        , keycache1(keycache1_)
        , keycache2(keycache2_) 
        , quantScale3(quantScale3_)
        , tiling(tiling_) {
    }
};

template <class BlockRmsNormAndRopeConvergence>
class MlaPreprocessKernel {
  public:
    using ArchTag = typename RmsNormAndRopeConvergence::ArchTag;
    using T = typename RmsNormAndRopeConvergence::T;

    // Methods
    CATLASS_DEVICE
    MlaPreprocessKernel() {
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(MlaPreprocessKernelParams const &params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(MlaPreprocessKernelParams const &params) {
        
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(MlaPreprocessKernelParams const &params) {
        // Get tiling parameters
        __gm__ MlaPreprocessTilingData *mpTilingData = reinterpret_cast<__gm__ MlaPreprocessTilingData *>(params.tiling);
        uint32_t rmsNumCol2 = mpTilingData->rmsNumCol2;

        // Get the memory offset address of the input on Global Memory
        // AscendC::GlobalTensor<T> hiddenStateGm;
        // hiddenStateGm.SetGlobalBuffer((__gm__ T *)params.hiddenState);
        AscendC::GlobalTensor<T> gamma3Gm;
        gamma3Gm.SetGlobalBuffer((__gm__ T *)params.gamma3);
        AscendC::GlobalTensor<T> slotMappingGm;
        slotMappingGm.SetGlobalBuffer((__gm__ T *)params.slotMapping);
        AscendC::GlobalTensor<T> descale1;
        descale1Gm.SetGlobalBuffer((__gm__ T *)params.descale1);
        AscendC::GlobalTensor<int8_t> s2Gm;
        s2Gm.SetGlobalBuffer((__gm__ int8_t *)params.s2);
        AscendC::GlobalTensor<int8_t> s3Gm;
        s3Gm.SetGlobalBuffer((__gm__ int8_t *)params.s3);
        AscendC::GlobalTensor<float> sin1Gm;
        sin1Gm.SetGlobalBuffer((__gm__ float *)params.sin1);
        AscendC::GlobalTensor<int8_t> s5Gm;
        s5Gm.SetGlobalBuffer((__gm__ int8_t *)params.s5);
        AscendC::GlobalTensor<float> cos1Gm;
        cos1Gm.SetGlobalBuffer((__gm__ float *)params.cos1);
        AscendC::GlobalTensor<float> keycache1;
        keycache1Gm.SetGlobalBuffer((__gm__ float *)params.keycache1);
        AscendC::GlobalTensor<float> keycache2;
        keycache2Gm.SetGlobalBuffer((__gm__ float *)params.keycache2);
        AscendC::GlobalTensor<float> quantScale3;
        quantScale3Gm.SetGlobalBuffer((__gm__ float *)params.quantScale3);

        BlockRmsNormAndRopeConvergence blockRmsNormAndRopeConvergence(resource);

        // Split core
        auto aicoreNum = AscendC::GetBlockNum();
        uint32_t row_work = (n + aicoreNum - 1) / aicoreNum; // 每个核处理多少行
        uint32_t need_core = (n + row_work - 1) / row_work; // 需要多少个核

        // Go through each task.
        uint32_t blockIdx = AscendC::GetBlockIdx();
        if (blockIdx < need_core) {
            uint32_t row_work_ = (blockIdx == need_core - 1) ? n - (need_core - 1) * row_work : row_work;
            BlockRmsNormAndRopeConvergence(
                row_work_, gamma3Gm, slotMappingGm, descale1Gm, rmsNumCol2, s2Gm, s3Gm, sin1Gm, 
                s5Gm, cos1Gm, keycache1Gm, keycache2Gm, quantScale3Gm
            );
        }
    }

  private:
    Arch::Resource<ArchTag> resource;
};

CATLASS_GLOBAL void MlaPreprocessFp16(
    uint64_t fftsAddr,
    GM_ADDR gamma3,
    GM_ADDR slotMapping,
    GM_ADDR descale1,
    GM_ADDR s2,
    GM_ADDR s3,
    GM_ADDR sin1,
    GM_ADDR s5,
    GM_ADDR cos1,
    GM_ADDR keycache1,
    GM_ADDR keycache2,
    GM_ADDR quantScale3,
    GM_ADDR tiling

) {
    AscendC::SetSyncBaseAddr(fftsAddr);

    using ArchTag = Arch::AtlasA2;
    using T = half;

    // GEMM Block模块，实现RmsNormQuant
    using BlockRmsNormAndRopeConvergence = Gemm::Block::RmsNormAndRopeConvergence<ArchTag, T, true, false, QuantMode::PER_TOKEN_SYMM_QUANT, false>;

    // Kernel level
    using MlaPreprocessKernel = MlaPreprocessKernel<BlockRmsNormAndRopeConvergence>;
    MlaPreprocessKernelParams params{gamma3, slotMapping, descale1, s2, s3, sin1, s5, cos1, keycache1, keycache2, quantScale3, tiling};

    // call kernel
    MlaPreprocessKernel mlaPreprocessKernel;
    mlaPreprocessKernel(params);
}

CATLASS_GLOBAL void MlaPreprocessBf16(
    uint64_t fftsAddr,
    GM_ADDR gamma3,
    GM_ADDR slotMapping,
    GM_ADDR descale1,
    GM_ADDR s2,
    GM_ADDR s3,
    GM_ADDR sin1,
    GM_ADDR s5,
    GM_ADDR cos1,
    GM_ADDR keycache1,
    GM_ADDR keycache2,
    GM_ADDR quantScale3,
    GM_ADDR tiling
) {
    AscendC::SetSyncBaseAddr(fftsAddr);

    using ArchTag = Arch::AtlasA2;
    using T = bfloat16_t;

    // GEMM Block模块，实现RmsNormQuant
    using BlockRmsNormAndRopeConvergence = Gemm::Block::RmsNormAndRopeConvergence<ArchTag, T, true, false, QuantMode::PER_TOKEN_SYMM_QUANT, false>;

    // Kernel level
    using MlaPreprocessKernel = MlaPreprocessKernel<BlockRmsNormQuant>;
    MlaPreprocessKernelParams params{gamma3, slotMapping, descale1, s2, s3, sin1, s5, cos1, keycache1, keycache2, quantScale3, tiling};

    // call kernel
    MlaPreprocessKernel mlaPreprocessKernel;
    mlaPreprocessKernel(params);
}.
