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
#include "catlass/gemm/block/block_rms_norm_quant.hpp"
#include "mla_preprocess_tiling.cpp"
using namespace Catlass;

struct MlaPreprocessKernelParams {
    GM_ADDR hiddenState;
    GM_ADDR gamma1;
    GM_ADDR beta1;
    GM_ADDR quantScale1;
    GM_ADDR quantOffset1;
    GM_ADDR descale1;
    GM_ADDR out;
    GM_ADDR scale;
    GM_ADDR tiling;
    // Methods
    CATLASS_DEVICE
    MlaPreprocessKernelParams() {
    }
    CATLASS_DEVICE
    MlaPreprocessKernelParams(GM_ADDR hiddenState_,
                    GM_ADDR gamma1_,
                    GM_ADDR beta1_,
                    GM_ADDR quantScale1_,
                    GM_ADDR quantOffset1_,
                    GM_ADDR descale1_,
                    GM_ADDR out_,
                    GM_ADDR scale_,
                    GM_ADDR tiling_)
        : hiddenState(hiddenState_)
        , gamma1(gamma1_)
        , beta1(beta1_)
        , quantScale1(quantScale1_)
        , quantOffset1(quantOffset1_)
        , descale1(descale1_)
        , out(out_)
        , scale(scale_)
        , tiling(tiling_) {
    }
};

template <class BlockRmsNormQuant>
class MlaPreprocessKernel {
  public:
    using ArchTag = typename BlockRmsNormQuant::ArchTag;
    using T = typename BlockRmsNormQuant::T;

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
        uint32_t n = mpTilingData->n;
        uint32_t he = mpTilingData->he;

        // Get the memory offset address of the input on Global Memory
        // AscendC::GlobalTensor<T> hiddenStateGm;
        // hiddenStateGm.SetGlobalBuffer((__gm__ T *)params.hiddenState);
        AscendC::GlobalTensor<T> gamma1Gm;
        gamma1Gm.SetGlobalBuffer((__gm__ T *)params.gamma1);
        AscendC::GlobalTensor<T> beta1Gm;
        beta1Gm.SetGlobalBuffer((__gm__ T *)params.beta1);
        AscendC::GlobalTensor<T> quantScale1Gm;
        quantScale1Gm.SetGlobalBuffer((__gm__ T *)params.quantScale1);
        AscendC::GlobalTensor<int8_t> quantOffset1Gm;
        quantOffset1Gm.SetGlobalBuffer((__gm__ int8_t *)params.quantOffset1);
        // AscendC::GlobalTensor<float> descale1Gm;
        // descale1Gm.SetGlobalBuffer((__gm__ float *)params.descale1);
        // AscendC::GlobalTensor<int8_t> outGm;
        // outGm.SetGlobalBuffer((__gm__ int8_t *)params.out);
        // AscendC::GlobalTensor<float> scaleGm;
        // scaleGm.SetGlobalBuffer((__gm__ float *)params.scale);
        
        BlockRmsNormQuant blockRmsNormQuant(resource, he);

        // Split core
        auto aicoreNum = AscendC::GetBlockNum();
        uint32_t row_work = (n + aicoreNum - 1) / aicoreNum; // 每个核处理多少行
        uint32_t need_core = (n + row_work - 1) / row_work; // 需要多少个核
//80行 40核 判断起始行
        // Go through each task.
        float avg_factor = 0.0001395089285f;
        uint32_t blockIdx = AscendC::GetBlockIdx();
        if (blockIdx < need_core) {
            uint32_t row_work_ = (blockIdx == need_core - 1) ? n - (need_core - 1) * row_work : row_work;
            blockRmsNormQuant(
                gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, params.scale + blockIdx * row_work * sizeof(float), 
                params.descale1, params.hiddenState, params.out, 0, he, avg_factor, blockIdx * row_work * he, 
                blockIdx * row_work * he, row_work_, n
            );
        }
    }

  private:
    Arch::Resource<ArchTag> resource;
};

CATLASS_GLOBAL void MlaPreprocessFp16(
    uint64_t fftsAddr,
    GM_ADDR hiddenState,
    GM_ADDR gamma1,
    GM_ADDR beta1,
    GM_ADDR quantScale1,
    GM_ADDR quantOffset1,
    GM_ADDR descale1,
    GM_ADDR out,
    GM_ADDR scale,
    GM_ADDR tiling
) {
    AscendC::SetSyncBaseAddr(fftsAddr);

    using ArchTag = Arch::AtlasA2;
    using T = half;

    // GEMM Block模块，实现RmsNormQuant
    using BlockRmsNormQuant = Gemm::Block::RmsNormQuant<ArchTag, T, true, false, QuantMode::PER_TOKEN_SYMM_QUANT, false>;

    // Kernel level
    using MlaPreprocessKernel = MlaPreprocessKernel<BlockRmsNormQuant>;
    MlaPreprocessKernelParams params{hiddenState, gamma1, beta1, quantScale1, quantOffset1, descale1, out, scale, tiling};

    // call kernel
    MlaPreprocessKernel mlaPreprocessKernel;
    mlaPreprocessKernel(params);
}

CATLASS_GLOBAL void MlaPreprocessBf16(
    uint64_t fftsAddr,
    GM_ADDR hiddenState,
    GM_ADDR gamma1,
    GM_ADDR beta1,
    GM_ADDR quantScale1,
    GM_ADDR quantOffset1,
    GM_ADDR descale1,
    GM_ADDR out,
    GM_ADDR scale,
    GM_ADDR tiling
) {
    AscendC::SetSyncBaseAddr(fftsAddr);

    using ArchTag = Arch::AtlasA2;
    using T = bfloat16_t;

    // GEMM Block模块，实现RmsNormQuant
    using BlockRmsNormQuant = Gemm::Block::RmsNormQuant<ArchTag, T, true, false, QuantMode::PER_TOKEN_SYMM_QUANT, false>;

    // Kernel level
    using MlaPreprocessKernel = MlaPreprocessKernel<BlockRmsNormQuant>;
    MlaPreprocessKernelParams params{hiddenState, gamma1, beta1, quantScale1, quantOffset1, descale1, out, scale, tiling};

    // call kernel
    MlaPreprocessKernel mlaPreprocessKernel;
    mlaPreprocessKernel(params);
}