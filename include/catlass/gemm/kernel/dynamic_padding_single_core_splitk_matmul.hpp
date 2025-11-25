/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_DYNAMIC_PADDING_SINGLE_CORE_SPLITK_MATMUL_HPP
#define CATLASS_GEMM_KERNEL_DYNAMIC_PADDING_SINGLE_CORE_SPLITK_MATMUL_HPP

#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/gemm/kernel/padding_matmul.hpp"
#include "catlass/gemm/helper.hpp"

namespace Catlass::Gemm::Kernel {

template <class PrologueA_, class PrologueB_, class BlockMmad_, 
    class BlockEpilogue_, class BlockScheduler_, class RemovePaddingC_>
class DynamicPaddingSingleCoreSplitkSimpleMatmul {
public:
    using PrologueA = PrologueA_;
    using PrologueB = PrologueB_;
    using RemovePaddingC = RemovePaddingC_;

    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using ElementA = typename BlockMmad::ElementA;
    using ElementB = typename BlockMmad::ElementB;
    using ElementC = typename BlockMmad::ElementC;

    using LayoutC = typename BlockMmad::LayoutC;

    template<class T>
    struct LayoutHelper {
        using type = typename T::LayoutIn;
    };
    template<>
    struct LayoutHelper<void> {
        using type = void;
    };

    using LayoutA = std::conditional_t<
        std::is_void_v<PrologueA>, typename BlockMmad::LayoutA, typename LayoutHelper<PrologueA>::type>;
    using LayoutB = std::conditional_t<
        std::is_void_v<PrologueB>, typename BlockMmad::LayoutB, typename LayoutHelper<PrologueB>::type>;

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
        GM_ADDR ptrWA;
        GM_ADDR ptrWB;
        GM_ADDR ptrWC;

        // Methods
        CATLASS_HOST_DEVICE
        Params()
        {}

        CATLASS_HOST_DEVICE
        Params(GemmCoord const &problemShape_, GemmCoord const &l1TileShape_, GemmCoord const &l0TileShape_, 
            GM_ADDR ptrA_, LayoutA& layoutA_, GM_ADDR ptrB_, LayoutB& layoutB_, GM_ADDR ptrC_, LayoutC& layoutC_,
            GM_ADDR ptrWA_, GM_ADDR ptrWB_, GM_ADDR ptrWC_)
            : problemShape(problemShape_), l1TileShape(l1TileShape_), l0TileShape(l0TileShape_), ptrA(ptrA_),
            layoutA(layoutA_), ptrB(ptrB_), layoutB(layoutB_), ptrC(ptrC_), layoutC(layoutC_), ptrWA(ptrWA_),
            ptrWB(ptrWB_), ptrWC(ptrWC_)
        {}
    };

    // Methods
    CATLASS_DEVICE
    DynamicPaddingSingleCoreSplitkSimpleMatmul() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params, Catlass::Arch::Resource<ArchTag> &resource);

    template<>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params, Catlass::Arch::Resource<ArchTag> &resource)
    {
         if constexpr (!std::is_void_v<PrologueA>) {
            AscendC::GlobalTensor<ElementA> gmA;
            AscendC::GlobalTensor<ElementA> gmWA;
            gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrA));
            gmWA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrWA));
            typename BlockMmad::LayoutA layoutWA;
            if constexpr (PrologueA::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_ND) {
                layoutWA = PrologueA::GetWorkspaceLayout(params.layoutA, 512 / sizeof(ElementA));
            } else if constexpr (PrologueA::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_BLOCK_ND) {
                layoutWA = PrologueA::GetWorkspaceLayout(params.layoutA, params.l1TileShape.m(), params.l1TileShape.k());
            } else if constexpr (PrologueA::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_NZ) {
                layoutWA = PrologueA::GetWorkspaceLayout(params.layoutA);
            }
            PrologueA prologueA(resource);
            prologueA(gmWA, gmA, layoutWA, params.layoutA);
        }

        if constexpr (!std::is_void_v<PrologueB>) {
            AscendC::GlobalTensor<ElementB> gmB;
            AscendC::GlobalTensor<ElementB> gmWB;
            gmB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrB));
            gmWB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrWB));
            typename BlockMmad::LayoutB layoutWB;
            if constexpr (PrologueB::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_ND) {
                layoutWB = PrologueB::GetWorkspaceLayout(params.layoutB, 512 / sizeof(ElementB));
            } else if constexpr (PrologueB::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_BLOCK_ND) {
                layoutWB = PrologueB::GetWorkspaceLayout(params.layoutB, params.l1TileShape.k(), params.l1TileShape.n());
            } else if constexpr (PrologueB::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_NZ) {
                layoutWB = PrologueB::GetWorkspaceLayout(params.layoutB);
            }
            PrologueB prologueB(resource);
            prologueB(gmWB, gmB, layoutWB, params.layoutB);
            // 0x0 synchronization control between AI Core
        }
        if constexpr (!std::is_void_v<PrologueA> || !std::is_void_v<PrologueB>) {
            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishPadding);
        }

        if constexpr (!std::is_void_v<RemovePaddingC>) {
            Catlass::Arch::CrossCoreWaitFlag(flagAicFinish);
            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
            AscendC::GlobalTensor<ElementC> gmC;
            AscendC::GlobalTensor<ElementC> gmWC;
            gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrC));
            gmWC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrWC));
            LayoutC layoutWC = RemovePaddingC::GetWorkspaceLayout(params.layoutC, 512 / sizeof(ElementC));
            RemovePaddingC removePaddingC(resource);
            removePaddingC(gmC, gmWC, params.layoutC, layoutWC);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    /// Executes matmul
    template<>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const &params, Catlass::Arch::Resource<ArchTag> &resource)
    {
        if constexpr (!std::is_void_v<PrologueA> || !std::is_void_v<PrologueB>) {
            Catlass::Arch::CrossCoreWaitFlag(flagAivFinishPadding);
        }

        typename BlockMmad::LayoutA layoutA;
        typename BlockMmad::LayoutB layoutB;
        typename BlockMmad::LayoutC layoutC;

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        if constexpr (std::is_void_v<PrologueA>) {
            gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
            layoutA = params.layoutA;
        } else {
            gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrWA);
            if constexpr (PrologueA::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_ND) {
                layoutA = PrologueA::GetWorkspaceLayout(params.layoutA, 512 / sizeof(ElementA));
            } else if constexpr (PrologueA::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_BLOCK_ND) {
                layoutA = PrologueA::GetWorkspaceLayout(params.layoutA, params.l1TileShape.m(), params.l1TileShape.k());
            } else if constexpr (PrologueA::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_NZ) {
                layoutA = PrologueA::GetWorkspaceLayout(params.layoutA);
            }
        }
        AscendC::GlobalTensor<ElementB> gmB;
        if constexpr (std::is_void_v<PrologueB>) {
            gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
            layoutB = params.layoutB;
        } else {
            gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrWB);
            if constexpr (PrologueB::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_ND) {
                layoutB = PrologueB::GetWorkspaceLayout(params.layoutB, 512 / sizeof(ElementB));
            } else if constexpr (PrologueB::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_BLOCK_ND) {
                layoutB = PrologueB::GetWorkspaceLayout(params.layoutB, params.l1TileShape.k(), params.l1TileShape.n());
            } else if constexpr (PrologueB::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_NZ) {
                layoutB = PrologueB::GetWorkspaceLayout(params.layoutB);
            }
        }
        AscendC::GlobalTensor<ElementC> gmC;
        if constexpr (std::is_void_v<RemovePaddingC>) {
            gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);
            layoutC = params.layoutC;
        } else {
            gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrWC);
            layoutC = RemovePaddingC::GetWorkspaceLayout(params.layoutC, 512 / sizeof(ElementC));
        }

        BlockMmad blockMmad(params.l1TileShape, params.l0TileShape, resource);

        BlockScheduler matmulBlockScheduler(params.problemShape, params.l1TileShape);
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
            int64_t gmOffsetA = layoutA.GetOffset(coordA);
            int64_t gmOffsetB = layoutB.GetOffset(coordB);
            int64_t gmOffsetC = layoutC.GetOffset(coordC);

            MatrixCoord coordNextA{
                nextBlockCoord.m() * params.l1TileShape.m(), nextBlockCoord.k() * params.l1TileShape.k()};
            MatrixCoord coordNextB{
                nextBlockCoord.k() * params.l1TileShape.k(), nextBlockCoord.n() * params.l1TileShape.n()};
            int64_t gmOffsetNextA = layoutA.GetOffset(coordNextA);
            int64_t gmOffsetNextB = layoutB.GetOffset(coordNextB);

            // Compute block-scoped matrix multiply-add
            blockMmad(gmA[gmOffsetA],
                layoutA,
                gmB[gmOffsetB],
                layoutB,
                gmC[gmOffsetC],
                layoutC,
                gmA[gmOffsetNextA],
                gmB[gmOffsetNextB],
                actualBlockShape,
                nextActualBlockShape,
                needLoadNextA,
                needLoadNextB,
                false);
        }
        if constexpr (!std::is_void_v<RemovePaddingC>) {
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flagAicFinish);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }
private:
    static constexpr Arch::FlagID FLAG_AIV_FINISH_STORE = 0;
    Arch::CrossCoreFlag flagAivFinishPadding{FLAG_AIV_FINISH_STORE};
    static constexpr Arch::FlagID FLAG_AIC_FINISH = 1;
    Arch::CrossCoreFlag flagAicFinish{FLAG_AIC_FINISH};
};

template <class PrologueA_, class PrologueB_, class BlockMmad_, 
    class BlockEpilogue_, class BlockScheduler_, class RemovePaddingNDAndCastC_>
class DynamicPaddingSingleCoreSplitkMatmul {
public:
    using PrologueA = PrologueA_;
    using PrologueB = PrologueB_;
    using RemovePaddingNDAndCastC = RemovePaddingNDAndCastC_;

    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using ElementA = typename BlockMmad::ElementA;
    using ElementB = typename BlockMmad::ElementB;

    template<class T>
    struct LayoutHelper {
        using type = typename T::LayoutIn;
    };
    template<>
    struct LayoutHelper<void> {
        using type = void;
    };

    using LayoutA = std::conditional_t<
        std::is_void_v<PrologueA>, typename BlockMmad::LayoutA, typename LayoutHelper<PrologueA>::type>;
    using LayoutB = std::conditional_t<
        std::is_void_v<PrologueB>, typename BlockMmad::LayoutB, typename LayoutHelper<PrologueB>::type>;

    template<class T>
    struct ElementHelper {
        using ElementC = typename T::ElementOut;
    };
    template<>
    struct ElementHelper<void> {
        using ElementC = typename BlockMmad::ElementC;
    };
    using ElementAccumulator = typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
    using ElementC = typename ElementHelper<RemovePaddingNDAndCastC>::ElementC;
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
        GM_ADDR ptrWA;
        GM_ADDR ptrWB;
        GM_ADDR ptrWC;

        // Methods
        CATLASS_HOST_DEVICE
        Params()
        {}

        CATLASS_HOST_DEVICE
        Params(GemmCoord const &problemShape_, GemmCoord const &l1TileShape_, GemmCoord const &l0TileShape_, 
            GM_ADDR ptrA_, LayoutA& layoutA_, GM_ADDR ptrB_, LayoutB& layoutB_, GM_ADDR ptrC_, LayoutC& layoutC_,
            GM_ADDR ptrWA_, GM_ADDR ptrWB_, GM_ADDR ptrWC_)
            : problemShape(problemShape_), l1TileShape(l1TileShape_), l0TileShape(l0TileShape_), ptrA(ptrA_),
            layoutA(layoutA_), ptrB(ptrB_), layoutB(layoutB_), ptrC(ptrC_), layoutC(layoutC_), ptrWA(ptrWA_),
            ptrWB(ptrWB_), ptrWC(ptrWC_)
        {}
    };

    // Methods
    CATLASS_DEVICE
    DynamicPaddingSingleCoreSplitkMatmul() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params, Catlass::Arch::Resource<ArchTag> &resource);

    template<>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params, Catlass::Arch::Resource<ArchTag> &resource)
    {
         if constexpr (!std::is_void_v<PrologueA>) {
            AscendC::GlobalTensor<ElementA> gmA;
            AscendC::GlobalTensor<ElementA> gmWA;
            gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrA));
            gmWA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrWA));
            typename BlockMmad::LayoutA layoutWA;
            if constexpr (PrologueA::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_ND) {
                layoutWA = PrologueA::GetWorkspaceLayout(params.layoutA, 512 / sizeof(ElementA));
            } else if constexpr (PrologueA::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_BLOCK_ND) {
                layoutWA = PrologueA::GetWorkspaceLayout(params.layoutA, params.l1TileShape.m(), params.l1TileShape.k());
            } else if constexpr (PrologueA::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_NZ) {
                layoutWA = PrologueA::GetWorkspaceLayout(params.layoutA);
            }
            PrologueA prologueA(resource);
            prologueA(gmWA, gmA, layoutWA, params.layoutA);
        }

        if constexpr (!std::is_void_v<PrologueB>) {
            AscendC::GlobalTensor<ElementB> gmB;
            AscendC::GlobalTensor<ElementB> gmWB;
            gmB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrB));
            gmWB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrWB));
            typename BlockMmad::LayoutB layoutWB;
            if constexpr (PrologueB::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_ND) {
                layoutWB = PrologueB::GetWorkspaceLayout(params.layoutB, 512 / sizeof(ElementB));
            } else if constexpr (PrologueB::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_BLOCK_ND) {
                layoutWB = PrologueB::GetWorkspaceLayout(params.layoutB, params.l1TileShape.k(), params.l1TileShape.n());
            } else if constexpr (PrologueB::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_NZ) {
                layoutWB = PrologueB::GetWorkspaceLayout(params.layoutB);
            }
            PrologueB prologueB(resource);
            prologueB(gmWB, gmB, layoutWB, params.layoutB);
            // 0x0 synchronization control between AI Core
        }
        if constexpr (!std::is_void_v<PrologueA> || !std::is_void_v<PrologueB>) {
            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishPadding);
        }

        if constexpr (!std::is_void_v<RemovePaddingNDAndCastC>) {
            Catlass::Arch::CrossCoreWaitFlag(flagAicFinish);
            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
            AscendC::GlobalTensor<ElementC> gmC;
            AscendC::GlobalTensor<ElementAccumulator> gmWC;
            gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrC));
            gmWC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementAccumulator *>(params.ptrWC));
            if constexpr (RemovePaddingNDAndCastC::paddingTag == PaddingTag::NO_PADDING) {
                RemovePaddingNDAndCastC removePaddingNDAndCastC(resource);
                removePaddingNDAndCastC(gmC, gmWC, params.layoutC, params.layoutC);
            } else {
                LayoutC layoutWC = RemovePaddingNDAndCastC::GetWorkspaceLayout(params.layoutC, 512 / sizeof(ElementAccumulator));
                RemovePaddingNDAndCastC removePaddingNDAndCastC(resource);
                removePaddingNDAndCastC(gmC, gmWC, params.layoutC, layoutWC);
            }
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    /// Executes matmul
    template<>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const &params, Catlass::Arch::Resource<ArchTag> &resource)
    {
        if constexpr (!std::is_void_v<PrologueA> || !std::is_void_v<PrologueB>) {
            Catlass::Arch::CrossCoreWaitFlag(flagAivFinishPadding);
        }

        typename BlockMmad::LayoutA layoutA;
        typename BlockMmad::LayoutB layoutB;
        typename BlockMmad::LayoutC layoutC;

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        if constexpr (std::is_void_v<PrologueA>) {
            gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
            layoutA = params.layoutA;
        } else {
            gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrWA);
            if constexpr (PrologueA::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_ND) {
                layoutA = PrologueA::GetWorkspaceLayout(params.layoutA, 512 / sizeof(ElementA));
            } else if constexpr (PrologueA::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_BLOCK_ND) {
                layoutA = PrologueA::GetWorkspaceLayout(params.layoutA, params.l1TileShape.m(), params.l1TileShape.k());
            } else if constexpr (PrologueA::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_NZ) {
                layoutA = PrologueA::GetWorkspaceLayout(params.layoutA);
            }
        }
        AscendC::GlobalTensor<ElementB> gmB;
        if constexpr (std::is_void_v<PrologueB>) {
            gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
            layoutB = params.layoutB;
        } else {
            gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrWB);
            if constexpr (PrologueB::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_ND) {
                layoutB = PrologueB::GetWorkspaceLayout(params.layoutB, 512 / sizeof(ElementB));
            } else if constexpr (PrologueB::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_BLOCK_ND) {
                layoutB = PrologueB::GetWorkspaceLayout(params.layoutB, params.l1TileShape.k(), params.l1TileShape.n());
            } else if constexpr (PrologueB::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_NZ) {
                layoutB = PrologueB::GetWorkspaceLayout(params.layoutB);
            }
        }
        AscendC::GlobalTensor<ElementAccumulator> gmC;
        if constexpr (!std::is_void_v<RemovePaddingNDAndCastC>) {
            gmC.SetGlobalBuffer((__gm__ ElementAccumulator *)params.ptrWC);
            if constexpr (RemovePaddingNDAndCastC::paddingTag == Catlass::Gemm::Kernel::PaddingTag::NO_PADDING) {
                layoutC = params.layoutC;
            } else {
                layoutC = RemovePaddingNDAndCastC::GetWorkspaceLayout(params.layoutC, 512 / sizeof(ElementAccumulator));
            }
        } else {
            gmC.SetGlobalBuffer((__gm__ ElementAccumulator *)params.ptrC);
            layoutC = params.layoutC;
        }

        BlockMmad blockMmad(params.l1TileShape, params.l0TileShape, resource);

        BlockScheduler matmulBlockScheduler(params.problemShape, params.l1TileShape);
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
            int64_t gmOffsetA = layoutA.GetOffset(coordA);
            int64_t gmOffsetB = layoutB.GetOffset(coordB);
            int64_t gmOffsetC = layoutC.GetOffset(coordC);

            MatrixCoord coordNextA{
                nextBlockCoord.m() * params.l1TileShape.m(), nextBlockCoord.k() * params.l1TileShape.k()};
            MatrixCoord coordNextB{
                nextBlockCoord.k() * params.l1TileShape.k(), nextBlockCoord.n() * params.l1TileShape.n()};
            int64_t gmOffsetNextA = layoutA.GetOffset(coordNextA);
            int64_t gmOffsetNextB = layoutB.GetOffset(coordNextB);

            // Compute block-scoped matrix multiply-add
            blockMmad(gmA[gmOffsetA],
                layoutA,
                gmB[gmOffsetB],
                layoutB,
                gmC[gmOffsetC],
                layoutC,
                gmA[gmOffsetNextA],
                gmB[gmOffsetNextB],
                actualBlockShape,
                nextActualBlockShape,
                needLoadNextA,
                needLoadNextB,
                matmulBlockScheduler.IsAtomicAdd(loopIdx));
        }
        if constexpr (!std::is_void_v<RemovePaddingNDAndCastC>) {
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flagAicFinish);
        }
        AscendC::SetAtomicNone();
        AscendC::PipeBarrier<PIPE_ALL>();
    }
private:
    static constexpr Arch::FlagID FLAG_AIV_FINISH_STORE = 0;
    Arch::CrossCoreFlag flagAivFinishPadding{FLAG_AIV_FINISH_STORE};
    static constexpr Arch::FlagID FLAG_AIC_FINISH = 1;
    Arch::CrossCoreFlag flagAicFinish{FLAG_AIC_FINISH};
};

template <
    class PrologueA_,
    class PrologueB_,
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_,
    class RemovePaddingNDAndCastC_>
class DynamicPaddingSingleCoreSplitkAsyncMatmul {
public:
    using PrologueA = PrologueA_;
    using PrologueB = PrologueB_;
    using RemovePaddingNDAndCastC = RemovePaddingNDAndCastC_;

    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using ElementA = typename BlockMmad::ElementA;
    using ElementB = typename BlockMmad::ElementB;

    template<class T>
    struct LayoutHelper {
        using type = typename T::LayoutIn;
    };
    template<>
    struct LayoutHelper<void> {
        using type = void;
    };

    using LayoutA = std::conditional_t<
        std::is_void_v<PrologueA>, typename BlockMmad::LayoutA, typename LayoutHelper<PrologueA>::type>;
    using LayoutB = std::conditional_t<
        std::is_void_v<PrologueB>, typename BlockMmad::LayoutB, typename LayoutHelper<PrologueB>::type>;

    template<class T>
    struct ElementHelper {
        using ElementC = typename T::ElementOut;
    };
    template<>
    struct ElementHelper<void> {
        using ElementC = typename BlockMmad::ElementC;
    };
    using ElementAccumulator = typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
    using ElementC = typename ElementHelper<RemovePaddingNDAndCastC>::ElementC;
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
        GM_ADDR ptrWA;
        GM_ADDR ptrWB;
        GM_ADDR ptrWC;

        // Methods
        CATLASS_HOST_DEVICE
        Params()
        {}

        CATLASS_HOST_DEVICE
        Params(GemmCoord const &problemShape_, GemmCoord const &l1TileShape_, GemmCoord const &l0TileShape_, 
            GM_ADDR ptrA_, LayoutA& layoutA_, GM_ADDR ptrB_, LayoutB& layoutB_, GM_ADDR ptrC_, LayoutC& layoutC_,
            GM_ADDR ptrWA_, GM_ADDR ptrWB_, GM_ADDR ptrWC_)
            : problemShape(problemShape_), l1TileShape(l1TileShape_), l0TileShape(l0TileShape_), ptrA(ptrA_),
            layoutA(layoutA_), ptrB(ptrB_), layoutB(layoutB_), ptrC(ptrC_), layoutC(layoutC_), ptrWA(ptrWA_),
            ptrWB(ptrWB_), ptrWC(ptrWC_)
        {}
    };

    // Methods
    CATLASS_DEVICE
    DynamicPaddingSingleCoreSplitkAsyncMatmul() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params, Catlass::Arch::Resource<ArchTag> &resource)
    {
        uint32_t aiCoreIdx;
        if ASCEND_IS_AIC {
            aiCoreIdx = AscendC::GetBlockIdx(); // cube:0-23
        } else if ASCEND_IS_AIV {
            aiCoreIdx = AscendC::GetBlockIdx() / 2; // vec:0-47 -> 0-23
        }

        typename BlockMmad::LayoutA layoutA;
        typename BlockMmad::LayoutB layoutB;
        typename BlockMmad::LayoutC layoutC;

        // AIV padding matrix A/B
        AscendC::GlobalTensor<ElementA> gmA, gmWA;
        AscendC::GlobalTensor<ElementB> gmB, gmWB;
        if constexpr (std::is_void_v<PrologueA>) {
            layoutA = params.layoutA;
            gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrA));
        } else {
            typename BlockMmad::LayoutA layoutWA;
            gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrA));
            gmWA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrWA));
            if constexpr (PrologueA::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_ND) {
                layoutWA = PrologueA::GetWorkspaceLayout(params.layoutA, 512 / sizeof(ElementA));
            } else if constexpr (PrologueA::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_BLOCK_ND) {
                layoutWA = PrologueA::GetWorkspaceLayout(params.layoutA, params.l1TileShape.m(), params.l1TileShape.k());
            } else if constexpr (PrologueA::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_NZ) {
                layoutWA = PrologueA::GetWorkspaceLayout(params.layoutA);
            }

            if ASCEND_IS_AIV {
                PrologueA prologueA(resource);
                prologueA(gmWA, gmA, layoutWA, params.layoutA);
            } else if ASCEND_IS_AIC {
                layoutA = layoutWA;
                gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrWA)); // after padding set gmA to gmWA
            }
        }
        if constexpr (std::is_void_v<PrologueB>) {
            layoutB = params.layoutB;
            gmB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrB));
        } else {
            typename BlockMmad::LayoutB layoutWB;
            gmB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrB));
            gmWB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrWB));
            if constexpr (PrologueB::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_ND) {
                layoutWB = PrologueB::GetWorkspaceLayout(params.layoutB, 512 / sizeof(ElementB));
            } else if constexpr (PrologueB::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_BLOCK_ND) {
                layoutWB = PrologueB::GetWorkspaceLayout(params.layoutB, params.l1TileShape.k(), params.l1TileShape.n());
            } else if constexpr (PrologueB::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_NZ) {
                layoutWB = PrologueB::GetWorkspaceLayout(params.layoutB);
            }

            if ASCEND_IS_AIV {
                PrologueB prologueB(resource);
                prologueB(gmWB, gmB, layoutWB, params.layoutB);
            } else if ASCEND_IS_AIC {
                gmB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrWB)); // after padding set gmB to gmWB
                layoutB = layoutWB;
            }
        }
        // matrix C
        AscendC::GlobalTensor<ElementAccumulator> gmWC;
        AscendC::GlobalTensor<ElementC> gmC;
        gmWC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementAccumulator *>(params.ptrWC));
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrC));
        layoutC = params.layoutC;
        RemovePaddingNDAndCastC removePaddingNDAndCastC(resource);

        // wait padding A/B finished
        if constexpr (!std::is_void_v<PrologueA> || !std::is_void_v<PrologueB>) {
            if ASCEND_IS_AIV {
                Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishPadding);
            } else if ASCEND_IS_AIC {
                Catlass::Arch::CrossCoreWaitFlag(flagAivFinishPadding);
            }
        }

        if ASCEND_IS_AIV {
            Catlass::Arch::CrossCoreSetFlag<0x02, PIPE_MTE3>(flagAivFinishMoveGmcIds[0]);
            Catlass::Arch::CrossCoreSetFlag<0x02, PIPE_MTE3>(flagAivFinishMoveGmcIds[1]);
        }
        BlockMmad blockMmad(params.l1TileShape, params.l0TileShape, resource);

        BlockScheduler matmulBlockScheduler(params.problemShape, params.l1TileShape);
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        GemmCoord baseBlockCoord, nextBaseBlockCoord;
        GemmCoord baseBlockShape, nextBaseBlockShape;
        GemmCoord baseOffsetCoord, nextBaseOffsetCoord;

        // LayoutWC RowMajor
        GemmCoord baseTile = matmulBlockScheduler.GetBaseTile();
        LayoutC layoutWC{baseTile.m(), baseTile.n(), baseTile.n()};
        uint64_t gmWcSliceSize = layoutWC.Capacity();

        // both AIC and AIV has this variable, and should update it separately
        uint32_t flagAivFinishMoveGmcBufId = 0;
        for (uint32_t loopIdx = aiCoreIdx; loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            bool isFirstBaseBlock = (loopIdx == aiCoreIdx);
            if (isFirstBaseBlock) {
                baseBlockCoord = matmulBlockScheduler.GetBaseBlockCoord(loopIdx);
                baseBlockShape = matmulBlockScheduler.GetBaseBlockShape(baseBlockCoord);
                baseOffsetCoord = matmulBlockScheduler.GetBaseOffsetCoord(baseBlockCoord);
            } else {
                baseBlockCoord = nextBaseBlockCoord;
                baseBlockShape = nextBaseBlockShape;
                baseOffsetCoord = nextBaseOffsetCoord;
            }

            bool hasNextBaseBlock = (loopIdx + AscendC::GetBlockNum() < coreLoops);
            if (hasNextBaseBlock) {
                nextBaseBlockCoord = matmulBlockScheduler.GetBaseBlockCoord(loopIdx + AscendC::GetBlockNum());
                nextBaseBlockShape = matmulBlockScheduler.GetBaseBlockShape(nextBaseBlockCoord);
                nextBaseOffsetCoord = matmulBlockScheduler.GetBaseOffsetCoord(nextBaseBlockCoord);
            }

            GemmCoord innerLoopsMNK = CeilDiv(baseBlockShape, params.l1TileShape);
            uint32_t innerLoops = innerLoopsMNK.m() * innerLoopsMNK.n() * innerLoopsMNK.k();

            MatrixCoord coordBaseBlock{baseOffsetCoord.m() * params.l1TileShape.m(), baseOffsetCoord.n() * params.l1TileShape.n()};
            uint64_t gmOffsetBaseBlock = layoutC.GetOffset(coordBaseBlock);
            LayoutC layoutInWC{baseBlockShape.m(), baseBlockShape.n(), baseTile.n()};
            int64_t gmOffsetWcSlice = (aiCoreIdx * GMWC_BUFFER_NUM + flagAivFinishMoveGmcBufId) * gmWcSliceSize;

            if ASCEND_IS_AIC {
                Catlass::Arch::CrossCoreWaitFlag(flagAivFinishMoveGmcIds[flagAivFinishMoveGmcBufId]);

                GemmCoord innerCoord;
                GemmCoord blockCoord, nextBlockCoord; // actual BlockCoord
                GemmCoord actualBlockShape, nextActualBlockShape;
                for (uint32_t innerLoopIdx = 0; innerLoopIdx < innerLoops; innerLoopIdx++) {
                    innerCoord = matmulBlockScheduler.GetInnerCoord(innerLoopsMNK, innerLoopIdx);
                    blockCoord = baseOffsetCoord + innerCoord;
                    actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

                    if (innerLoopIdx < innerLoops - 1) {
                        nextBlockCoord = baseOffsetCoord + matmulBlockScheduler.GetInnerCoord(innerLoopsMNK, innerLoopIdx + 1);
                    } else {
                        nextBlockCoord = nextBaseOffsetCoord;
                    }
                    nextActualBlockShape = matmulBlockScheduler.GetActualBlockShape(nextBlockCoord);

                    bool needLoadNextA = false, needLoadNextB = false;
                    if (hasNextBaseBlock || (innerLoopIdx < innerLoops - 1)) {
                        needLoadNextA = (blockCoord.k() != nextBlockCoord.k()) || (blockCoord.m() != nextBlockCoord.m());
                        needLoadNextB = (blockCoord.k() != nextBlockCoord.k()) || (blockCoord.n() != nextBlockCoord.n());
                    }

                    // Compute initial location in logical coordinates
                    MatrixCoord coordA{blockCoord.m() * params.l1TileShape.m(), blockCoord.k() * params.l1TileShape.k()};
                    MatrixCoord coordB{blockCoord.k() * params.l1TileShape.k(), blockCoord.n() * params.l1TileShape.n()};
                    int64_t gmOffsetA = layoutA.GetOffset(coordA);
                    int64_t gmOffsetB = layoutB.GetOffset(coordB);

                    // coordC is based on innerCoord, and gmOffsetC is in gmWC slice
                    MatrixCoord coordC{innerCoord.m() * params.l1TileShape.m(), innerCoord.n() * params.l1TileShape.n()};
                    int64_t gmOffsetC = layoutWC.GetOffset(coordC) + gmOffsetWcSlice;

                    MatrixCoord coordNextA{
                        nextBlockCoord.m() * params.l1TileShape.m(), nextBlockCoord.k() * params.l1TileShape.k()};
                    MatrixCoord coordNextB{
                        nextBlockCoord.k() * params.l1TileShape.k(), nextBlockCoord.n() * params.l1TileShape.n()};
                    int64_t gmOffsetNextA = layoutA.GetOffset(coordNextA);
                    int64_t gmOffsetNextB = layoutB.GetOffset(coordNextB);

                    bool isAtomicAdd = (blockCoord.k() > 0);

                    // Compute block-scoped matrix multiply-add
                    blockMmad(gmA[gmOffsetA],
                        layoutA,
                        gmB[gmOffsetB],
                        layoutB,
                        gmWC[gmOffsetC],
                        layoutWC,
                        gmA[gmOffsetNextA],
                        gmB[gmOffsetNextB],
                        actualBlockShape,
                        nextActualBlockShape,
                        needLoadNextA,
                        needLoadNextB,
                        isAtomicAdd);
                } // end inner loop

                Catlass::Arch::CrossCoreSetFlag<0x02, PIPE_FIX>(flagAicFinish);
                flagAivFinishMoveGmcBufId = (flagAivFinishMoveGmcBufId + 1) % GMWC_BUFFER_NUM; // update in AIC
            } else if ASCEND_IS_AIV {
                Catlass::Arch::CrossCoreWaitFlag(flagAicFinish);

                constexpr bool useSingleCore = true; // only use 2 vector of current aicore
                removePaddingNDAndCastC(gmC[gmOffsetBaseBlock], gmWC[gmOffsetWcSlice], params.layoutC, layoutInWC, useSingleCore);

                Catlass::Arch::CrossCoreSetFlag<0x02, PIPE_MTE3>(flagAivFinishMoveGmcIds[flagAivFinishMoveGmcBufId]);
                flagAivFinishMoveGmcBufId = (flagAivFinishMoveGmcBufId + 1) % GMWC_BUFFER_NUM; // update in AIV
            }
        }
        if ASCEND_IS_AIC {
            Catlass::Arch::CrossCoreWaitFlag(flagAivFinishMoveGmcIds[0]);
            Catlass::Arch::CrossCoreWaitFlag(flagAivFinishMoveGmcIds[1]);
            AscendC::SetAtomicNone();
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    static constexpr Arch::FlagID FLAG_AIV_FINISH_STORE = 0;
    Arch::CrossCoreFlag flagAivFinishPadding{FLAG_AIV_FINISH_STORE};
    static constexpr Arch::FlagID FLAG_AIC_FINISH = 1;
    Arch::CrossCoreFlag flagAicFinish{FLAG_AIC_FINISH};
    static constexpr uint32_t GMWC_BUFFER_NUM = 2;
    Arch::CrossCoreFlag flagAivFinishMoveGmcIds[GMWC_BUFFER_NUM] = {Arch::FlagID{2}, Arch::FlagID{3}};
};

}  // namespace Catlass::Gemm::Kernel

#endif  // CATLASS_GEMM_KERNEL_DYNAMIC_PADDING_SINGLE_CORE_SPLITK_MATMUL_HPP
