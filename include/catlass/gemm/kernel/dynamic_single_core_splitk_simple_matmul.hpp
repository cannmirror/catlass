/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_DYNAMIC_SINGLE_CORE_SPLITK_SIMPLE_MATMUL_HPP
#define CATLASS_GEMM_KERNEL_DYNAMIC_SINGLE_CORE_SPLITK_SIMPLE_MATMUL_HPP

#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/gemm/helper.hpp"

namespace Catlass::Gemm::Kernel {

template <class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class DynamicSingleCoreSplitkSimpleMatmul {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using ElementA = typename BlockMmad::ElementA;
    using ElementB = typename BlockMmad::ElementB;
    using ElementC = typename BlockMmad::ElementC;

    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;

    using BlockScheduler = BlockScheduler_;

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;
        GemmCoord l1TileShape;
        GemmCoord l0TileShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrC;
        LayoutC layoutC;
        uint32_t swizzleOffset;
        uint32_t swizzleDirection;

        // Methods
        CATLASS_HOST_DEVICE
        Params()
        {}

        CATLASS_HOST_DEVICE
        Params(GemmCoord const &problemShape_, GemmCoord const &l1TileShape_, GemmCoord const &l0TileShape_, 
            GM_ADDR ptrA_, LayoutA& layoutA_, GM_ADDR ptrB_, LayoutB& layoutB_, GM_ADDR ptrC_, LayoutC& layoutC_,
            uint32_t swizzleOffset_, uint32_t swizzleDirection_)
            : problemShape(problemShape_), l1TileShape(l1TileShape_), l0TileShape(l0TileShape_), ptrA(ptrA_),
            layoutA(layoutA_), ptrB(ptrB_), layoutB(layoutB_), ptrC(ptrC_), layoutC(layoutC_),
            swizzleOffset(swizzleOffset_), swizzleDirection(swizzleDirection_)
        {}
    };

    // Methods
    CATLASS_DEVICE
    DynamicSingleCoreSplitkSimpleMatmul() {}

    CATLASS_DEVICE
    void operator()(Params const &params, Catlass::Arch::Resource<ArchTag> &resource)
    {
        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);

        BlockMmad blockMmad(params.l1TileShape, params.l0TileShape, resource);

        BlockScheduler matmulBlockScheduler(params.problemShape, params.l1TileShape, params.swizzleOffset, params.swizzleDirection);
        uint32_t coreLoops = matmulBlockScheduler.GetSingleCoreLoops();

        GemmCoord blockCoord;
        GemmCoord actualBlockShape;
        GemmCoord nextBlockCoord;
        GemmCoord nextActualBlockShape;

        for (uint32_t loopIdx = 0; loopIdx < coreLoops; loopIdx++) {

            bool isFirstBlock = (loopIdx == 0);
            if (isFirstBlock) {
                blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
                actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);
            } else {
                blockCoord = nextBlockCoord;
                actualBlockShape = nextActualBlockShape;
            }

            // when loopIdx < coreLoops - 1, at least one of needLoadNextA/needLoadNextB is true
            // when loopIdx == coreLoops - 1, both are false
            bool needLoadNextA = false, needLoadNextB = false;
            if (loopIdx < coreLoops - 1) {
                nextBlockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx + 1);
                nextActualBlockShape = matmulBlockScheduler.GetActualBlockShape(nextBlockCoord);
                needLoadNextA = (blockCoord.k() != nextBlockCoord.k()) || (blockCoord.m() != nextBlockCoord.m());
                needLoadNextB = (blockCoord.k() != nextBlockCoord.k()) || (blockCoord.n() != nextBlockCoord.n());
            }

            // Compute initial location in logical coordinates
            MatrixCoord coordA{blockCoord.m() * params.l1TileShape.m(), blockCoord.k() * params.l1TileShape.k()};
            MatrixCoord coordB{blockCoord.k() * params.l1TileShape.k(), blockCoord.n() * params.l1TileShape.n()};
            MatrixCoord coordC{blockCoord.m() * params.l1TileShape.m(), blockCoord.n() * params.l1TileShape.n()};
            int64_t gmOffsetA = params.layoutA.GetOffset(coordA);
            int64_t gmOffsetB = params.layoutB.GetOffset(coordB);
            int64_t gmOffsetC = params.layoutC.GetOffset(coordC);

            MatrixCoord coordNextA{
                nextBlockCoord.m() * params.l1TileShape.m(), nextBlockCoord.k() * params.l1TileShape.k()};
            MatrixCoord coordNextB{
                nextBlockCoord.k() * params.l1TileShape.k(), nextBlockCoord.n() * params.l1TileShape.n()};
            int64_t gmOffsetNextA = params.layoutA.GetOffset(coordNextA);
            int64_t gmOffsetNextB = params.layoutB.GetOffset(coordNextB);

            // Compute block-scoped matrix multiply-add
            blockMmad(gmA[gmOffsetA],
                params.layoutA,
                gmB[gmOffsetB],
                params.layoutB,
                gmC[gmOffsetC],
                params.layoutC,
                gmA[gmOffsetNextA],
                gmB[gmOffsetNextB],
                actualBlockShape,
                nextActualBlockShape,
                needLoadNextA,
                needLoadNextB,
                false);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }
};

}  // namespace Catlass::Gemm::Kernel

#endif  // CATLASS_GEMM_KERNEL_DYNAMIC_SINGLE_CORE_SPLITK_SIMPLE_MATMUL_HPP