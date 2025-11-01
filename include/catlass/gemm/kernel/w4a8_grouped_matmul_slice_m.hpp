/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_W4A8_GROUPED_MATMUL_SLICE_M_HPP
#define CATLASS_GEMM_KERNEL_W4A8_GROUPED_MATMUL_SLICE_M_HPP

#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/epilogue/tile/copy_gm_to_ub.hpp"
#include "catlass/epilogue/tile/copy_ub_to_gm.hpp"
#include "catlass/gemm/kernel/padding_matmul.hpp"

inline __gm__ struct OpSystemRunCfg g_opSystemRunCfg{Catlass::L2_OFFSET};

namespace Catlass::Gemm::Kernel {

template <
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_,
    class ElementGroupList_
>
class W4A8GroupedMatmulSliceM {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using ElementA = typename BlockMmad::ElementA;
    using ElementPrologueB = typename BlockMmad::PrologueB::ElementSrc;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutPrologueB = typename BlockMmad::PrologueB::LayoutSrc;
    using LayoutB = typename BlockMmad::LayoutB;

    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;

    using MmadParams = typename BlockMmad::Params;

    using BlockScheduler = BlockScheduler_;

    using ElementGroupList = ElementGroupList_;

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;
        uint32_t problemCount;
        GM_ADDR ptrGroupList;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrPrologueB;
        LayoutPrologueB layoutPrologueB;
        GM_ADDR ptrC;
        LayoutC layoutC;

        GM_ADDR ptrScales;

        MmadParams mmadParams;

        GM_ADDR ptrWorkspace;

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(
            GemmCoord const &problemShape_, uint32_t problemCount_, GM_ADDR ptrGroupList_,
            GM_ADDR ptrA_, LayoutA const &layoutA_,
            GM_ADDR ptrPrologueB_, LayoutPrologueB const &layoutPrologueB_,
            GM_ADDR ptrC_, LayoutC const &layoutC_,
            GM_ADDR ptrScales_,
            MmadParams const &mmadParams_,
            GM_ADDR ptrWorkspace_
        ):  problemShape(problemShape_), problemCount(problemCount_), ptrGroupList(ptrGroupList_),
            ptrA(ptrA_), layoutA(layoutA_),
            ptrPrologueB(ptrPrologueB_), layoutPrologueB(layoutPrologueB_),
            ptrC(ptrC_), layoutC(layoutC_),
            ptrScales(ptrScales_),
            mmadParams(mmadParams_),
            ptrWorkspace(ptrWorkspace_) {}
    };

    struct Arguments {
        GemmCoord problemShape;
        uint32_t problemCount;
        GM_ADDR ptrGroupList;
        GM_ADDR deviceA;
        LayoutA layoutA;
        GM_ADDR devicePrologueB;
        LayoutPrologueB layoutPrologueB;
        GM_ADDR deviceC;
        LayoutC layoutC;
        GM_ADDR deviceScales;
        uint32_t aicoreNum;
    };

    static bool CanImplement(const Arguments &args)
    {
        return true;
    }

    static size_t GetWorkspaceSize(Arguments const &args)
    {
        return BlockMmad::STAGES * L1TileShape::K * L1TileShape::N * sizeof(ElementB) * args.aicoreNum;
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
    {
        Params params{
            args.problemShape, args.problemCount, args.ptrGroupList,
            args.deviceA, args.layoutA,
            args.devicePrologueB, args.layoutPrologueB,
            args.deviceC, args.layoutC,
            args.deviceScales,
            {{}, {}, {1.0}},
            workspace
        };
        return params;
    }

    // Methods
    CATLASS_DEVICE
    W4A8GroupedMatmulSliceM() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    /// Executes matmul
    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params)
    {
        auto aicoreNum = AscendC::GetBlockNum();
        auto aicoreIdx = AscendC::GetBlockIdx();

        int64_t gmGroupOffsetA = 0;
        int64_t gmGroupOffsetC = 0;

        BlockMmad blockMmad(resource, params.mmadParams);
        BlockScheduler matmulBlockScheduler;

        GemmCoord blockShape = L1TileShape::ToCoord();

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrA));
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrC));
        AscendC::GlobalTensor<ElementGroupList> gmGroupList;
        gmGroupList.SetGlobalBuffer(reinterpret_cast<__gm__ ElementGroupList *>(params.ptrGroupList));
        AscendC::GlobalTensor<float> gmScales;
        gmScales.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(params.ptrScales));

        LayoutB layoutBlockB{L1TileShape::K, L1TileShape::N};
        auto gmOffsetB = aicoreIdx * layoutBlockB.Capacity() * BlockMmad::STAGES;
        AscendC::GlobalTensor<ElementB> gmBlockB;
        gmBlockB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrWorkspace) + gmOffsetB);

        uint32_t startCoreIdx = 0;
        for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
            uint32_t currentM = (groupIdx == 0) ? gmGroupList.GetValue(groupIdx) :
                (gmGroupList.GetValue(groupIdx) - gmGroupList.GetValue(groupIdx - 1));
            GemmCoord inGroupProblemShape{currentM, params.problemShape.n(), params.problemShape.k()};

            float scale = gmScales.GetValue(groupIdx);
            blockMmad.UpdateParams({{}, {}, {scale}});

            LayoutA layoutA = params.layoutA.GetTileLayout(inGroupProblemShape.GetCoordMK());
            LayoutC layoutC = params.layoutC.GetTileLayout(inGroupProblemShape.GetCoordMN());

            matmulBlockScheduler.Update(inGroupProblemShape, blockShape.GetCoordMN());
            uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

            if (CeilDiv(currentM, L1TileShape::M) == 1) {
                gmBlockB.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            }

            uint32_t startLoopIdx;
            if (aicoreIdx < startCoreIdx) {
                startLoopIdx = aicoreIdx + aicoreNum - startCoreIdx;
            } else {
                startLoopIdx = aicoreIdx - startCoreIdx;
            }

            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += aicoreNum) {
                auto blockIdxCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
                auto actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockIdxCoord);
                GemmCoord offsetCoord = blockIdxCoord * blockShape;

                int64_t gmOffsetA = layoutA.GetOffset(offsetCoord.GetCoordMK());
                int64_t gmOffsetC = layoutC.GetOffset(offsetCoord.GetCoordMN());
                auto layoutBlockA = layoutA.GetTileLayout(actualBlockShape.GetCoordMK());
                auto layoutBlockC = layoutC.GetTileLayout(actualBlockShape.GetCoordMN());

                auto gmBlockA = gmA[params.layoutA.GetOffset(offsetCoord.GetCoordMK())];
                auto gmBlockC = gmC[params.layoutC.GetOffset(offsetCoord.GetCoordMN())];

                blockMmad(
                    gmA[gmGroupOffsetA + gmOffsetA], layoutBlockA,
                    gmBlockB, layoutBlockB,
                    gmC[gmGroupOffsetC + gmOffsetC], layoutBlockC,
                    actualBlockShape
                );
            }

            gmGroupOffsetA += inGroupProblemShape.m() * inGroupProblemShape.k();
            gmGroupOffsetC += inGroupProblemShape.m() * inGroupProblemShape.n();

            startCoreIdx = (startCoreIdx + coreLoops) % aicoreNum;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params)
    {
        auto aicoreNum = AscendC::GetBlockNum();
        auto aicoreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();

        int64_t gmGroupOffsetPrologueB = 0;
        GemmCoord blockShape = L1TileShape::ToCoord();

        BlockMmad blockMmad(resource, params.mmadParams);
        BlockScheduler matmulBlockScheduler;

        AscendC::GlobalTensor<ElementGroupList> gmGroupList;
        gmGroupList.SetGlobalBuffer(reinterpret_cast<__gm__ ElementGroupList *>(params.ptrGroupList));
        
        AscendC::GlobalTensor<ElementPrologueB> gmPrologueB;
        gmPrologueB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementPrologueB *>(params.ptrPrologueB));

        LayoutB layoutBlockB{L1TileShape::K, L1TileShape::N};
        auto gmOffsetB = aicoreIdx * layoutBlockB.Capacity() * BlockMmad::STAGES;
        AscendC::GlobalTensor<ElementB> gmBlockB;
        gmBlockB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrWorkspace) + gmOffsetB);

        uint32_t startCoreIdx = 0;
        for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
            uint32_t currentM = (groupIdx == 0) ? gmGroupList.GetValue(groupIdx) :
                (gmGroupList.GetValue(groupIdx) - gmGroupList.GetValue(groupIdx - 1));
            GemmCoord inGroupProblemShape{currentM, params.problemShape.n(), params.problemShape.k()};

            matmulBlockScheduler.Update(inGroupProblemShape, blockShape.GetCoordMN());
            uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();
            
            uint32_t startLoopIdx;
            if (aicoreIdx < startCoreIdx) {
                startLoopIdx = aicoreIdx + aicoreNum - startCoreIdx;
            } else {
                startLoopIdx = aicoreIdx - startCoreIdx;
            }

            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += aicoreNum) {
                // Compute block location
                auto blockIdxCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
                auto actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockIdxCoord);

                auto offsetCoordB = blockIdxCoord.GetCoordKN() * L1TileShape::ToCoordKN();
                auto gmOffsetPrologueB = params.layoutPrologueB.GetOffset(offsetCoordB);
                auto layoutBlockPrologueB = params.layoutPrologueB.GetTileLayout(actualBlockShape.GetCoordKN());

                // Compute block-scoped matrix multiply-add
                blockMmad.Prologue(
                    gmPrologueB[gmGroupOffsetPrologueB + gmOffsetPrologueB], layoutBlockPrologueB,
                    gmBlockB, layoutBlockB,
                    actualBlockShape
                );
            }
            gmGroupOffsetPrologueB += params.problemShape.n() * params.problemShape.k();
        }
    }

private:
    Arch::Resource<ArchTag> resource;
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_W4A8_GROUPED_MATMUL_SLICE_M_HPP