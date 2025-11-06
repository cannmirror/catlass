/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMMON_MATMUL_KERNEL_H
#define COMMON_MATMUL_KERNEL_H

#include "tiling_params.h"
#include "acl/acl.h"
#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/kernel/dynamic_common_matmul.hpp"
#include "catlass/gemm/gemm_type.hpp"

template <
    /// Tag indicating architecture
    class ArchTag,
    /// GemmType for A matrix operand
    class AType,
    /// GemmType type for B matrix operand
    class BType,
    /// GemmType type for C matrix operand
    class CType,
    /// GemmType type for Bias operand
    class BiasType = void
>
struct TileCopyDynamicOptimized : public Catlass::Gemm::Tile::TileCopy<ArchTag, AType, BType, CType, BiasType> {
    using CopyGmToL1A = typename Catlass::Gemm::Tile::CopyGmToL1DynamicOptimized<ArchTag, AType>;
    using CopyGmToL1B = typename Catlass::Gemm::Tile::CopyGmToL1DynamicOptimized<ArchTag, BType>;
};

template <class ArchTag, class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC>
CATLASS_GLOBAL __attribute__((aic)) void CommonMatmulKernel(__gm__ uint8_t *__restrict__ gmA,
    __gm__ uint8_t *__restrict__ gmB, __gm__ uint8_t *__restrict__ gmC, __gm__ uint8_t *__restrict__ tilingData)
{
    Catlass::Arch::Resource<ArchTag> resource;

    /*
    * Load tiling parameters from global memory (tilingData) to local array tilingParams
    * 
    * tilingData memory layout corresponds to tilingParams as follows:
    * --------------------------------------------------------------------------------
    * | Offset | Size | Variable         | Type      | Description                   |
    * |--------|------|------------------|-----------|-------------------------------|
    * | 0-7    | 8    | strideA          | uint64_t  | matrix B stride               |
    * | 8-15   | 8    | strideB          | uint64_t  | matrix B stride               |
    * | 16-23  | 8    | strideC          | uint64_t  | matrix C stride               |
    * | 24-27  | 4    | m                | uint32_t  | matrix M dimension            |
    * | 28-31  | 4    | n                | uint32_t  | matrix N dimension            |
    * | 32-35  | 4    | k                | uint32_t  | matrix K dimension            |
    * | 36-37  | 2    | m1               | uint16_t  | l1 mTile(16-bit to save space)|
    * | 38-39  | 2    | n1               | uint16_t  | l1 nTile(16-bit to save space)|
    * | 40-41  | 2    | k1               | uint16_t  | l1 kTile(16-bit to save space)|
    * | 42-42  | 1    | swizzleOffset    | uint8_t   | swizzle offset                |
    * | 43-43  | 1    | swizzleDirection | uint8_t   | swizzle direction             |
    * | 44-47  | 4    | (reserved)       | -         | unused                        |
    * --------------------------------------------------------------------------------
    */
    uint8_t tilingParams[48];
    // Copy data in 64-bit chunks to tilingParams array for efficiency
    // Copy bytes 0-7: strideA
    *(uint64_t *)(tilingParams) = *(reinterpret_cast<__gm__ uint64_t *>(tilingData));
    // Copy bytes 8-15: strideB
    *(uint64_t *)(tilingParams + 8) = *(reinterpret_cast<__gm__ uint64_t *>(tilingData + 8));
    // Copy bytes 16-23: strideC
    *(uint64_t *)(tilingParams + 16) = *(reinterpret_cast<__gm__ uint64_t *>(tilingData + 16));
    // Copy bytes 24-31: m, n
    *(uint64_t *)(tilingParams + 24) = *(reinterpret_cast<__gm__ uint64_t *>(tilingData + 24));
    // Copy bytes 32-39: k, m1, n1
    *(uint64_t *)(tilingParams + 32) = *(reinterpret_cast<__gm__ uint64_t *>(tilingData + 32));
    // Copy bytes 40-47: k1, swizzleOffset, swizzleDirection
    *(uint64_t *)(tilingParams + 40) = *(reinterpret_cast<__gm__ uint64_t *>(tilingData + 40));

    // read strideA: tilingParams[0:7]
    int64_t strideA = static_cast<int64_t>(*(reinterpret_cast<uint64_t *>(tilingParams)));
    // read strideB: tilingParams[8:15]
    int64_t strideB = static_cast<int64_t>(*(reinterpret_cast<uint64_t *>(tilingParams + 8)));
    // read strideC: tilingParams[16:23]
    int64_t strideC = static_cast<int64_t>(*(reinterpret_cast<uint64_t *>(tilingParams + 16)));
    // read m: tilingParams[24:27]
    uint32_t m = *(reinterpret_cast<uint32_t *>(tilingParams + 24));
    // read n: tilingParams[28:31]
    uint32_t n = *(reinterpret_cast<uint32_t *>(tilingParams + 28));
    // read k: tilingParams[32:35]
    uint32_t k = *(reinterpret_cast<uint32_t *>(tilingParams + 32));

    // To save space, tiling parameters (m1, n1, k1) are stored as uint16_t.
    // read m1: tilingParams[36:37]
    uint32_t m1 = *(reinterpret_cast<uint16_t *>(tilingParams + 36));
    // read n1: tilingParams[38:39]
    uint32_t n1 = *(reinterpret_cast<uint16_t *>(tilingParams + 38));
    // read k1: tilingParams[40:41]
    uint32_t k1 = *(reinterpret_cast<uint16_t *>(tilingParams + 40));

    // read swizzleOffset: tilingParams[42:42]
    uint32_t swizzleOffset = *(reinterpret_cast<uint8_t *>(tilingParams + 42));
    // read swizzleDirection: tilingParams[43:43]
    uint32_t swizzleDirection = *(reinterpret_cast<uint8_t *>(tilingParams + 43));

    Catlass::GemmCoord problemShape(m, n, k);
    Catlass::GemmCoord l1TileShape(m1, n1, k1);
    LayoutA layoutA{m, k, strideA};
    LayoutB layoutB{k, n, strideB};
    LayoutC layoutC{m, n, strideC};
    constexpr bool enableUnitFlag = true;
    constexpr bool enableShuffleK = true;
    using DispatchPolicy = Catlass::Gemm::MmadAtlasA2DynamicCommon<enableShuffleK, enableShuffleK>;

    using AType = Catlass::Gemm::GemmType<ElementA, LayoutA>;
    using BType = Catlass::Gemm::GemmType<ElementB, LayoutB>;
    using CType = Catlass::Gemm::GemmType<ElementC, LayoutC>;

    using TileCopy = TileCopyDynamicOptimized<ArchTag, AType, BType, CType>;
    using BlockMmad = Catlass::Gemm::Block::BlockMmad<DispatchPolicy, void, void, AType, BType, CType, void, TileCopy>;
    using BlockEpilogue = void;

    using BlockScheduler = typename Catlass::Gemm::Block::DynamicGemmIdentityBlockSwizzle;
    // kernel level
    using MatmulKernel = Catlass::Gemm::Kernel::DynamicCommonMatmul<BlockMmad, BlockEpilogue, BlockScheduler>;
    typename MatmulKernel::Params params{
        problemShape, l1TileShape, gmA, layoutA, gmB, layoutB, gmC, layoutC, swizzleOffset, swizzleDirection};
    // call a kernel
    MatmulKernel matmul;
    matmul(params, resource);
}

template <class ArchTag, class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC>
void LaunchCommonMatmulKernel(aclrtStream &stream, uint64_t fftsAddr, uint8_t *dA, uint8_t *dB, uint8_t *dC,
    uint8_t *dTilingParams, TilingParams &tilingParams)
{
    CommonMatmulKernel<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC>
        <<<tilingParams.blockDim, nullptr, stream>>>(dA, dB, dC, dTilingParams);
}

template <class ArchTag, class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC>
size_t CommonMatmulKernelGetWorkspaceSize(TilingParams &tilingParams)
{
    return 0;
}

#endif  // COMMON_MATMUL_KERNEL_H