/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_MATMUL_VISITOR_PHASED_HPP
#define CATLASS_GEMM_KERNEL_MATMUL_VISITOR_PHASED_HPP

#include <tuple>
#include <utility>

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

namespace Catlass::Gemm::Kernel {

template <
    class BlockMmad_,
    class BlockScheduler_,
    class... BlockEpilogues_
>
class MatmulVisitorPhased {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;

    using BlockScheduler = BlockScheduler_;

    using EpilogueTuple = std::tuple<BlockEpilogues_...>;
    static constexpr size_t NumPhases = std::tuple_size<EpilogueTuple>::value;

    using FirstEpilogue = std::tuple_element_t<0, EpilogueTuple>;
    using ElementC = typename FirstEpilogue::ElementC;
    using LayoutC = typename FirstEpilogue::LayoutC;

    struct ParamsTupleBuilder {
        template <class Epilogue>
        using ParamT = typename Epilogue::Params;
        template <class... Epilogues>
        using TupleT = std::tuple<ParamT<Epilogues>...>;
    };

    using EpilogueParamsTuple = tla::tuple<typename BlockEpilogues_::Params...>;

    struct Params {
        GemmCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrWorkspace; // C workspace base
        EpilogueParamsTuple epilogueParams;

        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(
            GemmCoord const& problemShape_,
            GM_ADDR ptrA_, LayoutA const& layoutA_,
            GM_ADDR ptrB_, LayoutB const& layoutB_,
            GM_ADDR ptrWorkspace_, EpilogueParamsTuple const& epilogueParams_)
            : problemShape(problemShape_), ptrA(ptrA_), layoutA(layoutA_), ptrB(ptrB_),
              layoutB(layoutB_), ptrWorkspace(ptrWorkspace_), epilogueParams(epilogueParams_) {}
    };

    // Build EVG args tuple for arbitrary number of phases
    template <class IndexSeq>
    struct EVGArgsTupleHelper;

    template <size_t... Is>
    struct EVGArgsTupleHelper<std::index_sequence<Is...>> {
        using type = std::tuple<typename std::tuple_element_t<Is, EpilogueTuple>::EVG::Arguments...>;
    };

    using EVGArgsTuple = typename EVGArgsTupleHelper<std::make_index_sequence<NumPhases>>::type;

    struct Arguments {
        GemmCoord problemShape;
        GM_ADDR ptrA;
        GM_ADDR ptrB;
        EVGArgsTuple evg_args;
    };

private:
    // Helper to build Arguments::evg tuple for arbitrary NumPhases
    template <size_t... Is>
    struct ArgumentsImpl {
        using TupleT = std::tuple<typename std::tuple_element_t<Is, EpilogueTuple>::EVG::Arguments...>;
    };

public:
    static bool CanImplement(Arguments const& args)
    {
        return can_implement_impl(args, std::make_index_sequence<NumPhases>{});
    }

    static size_t GetWorkspaceSize(Arguments const& args)
    {
        size_t bytes = sizeof(ElementC) * static_cast<size_t>(args.problemShape.m()) * args.problemShape.n();
        bytes += workspace_size_impl(args, std::make_index_sequence<NumPhases>{});
        return bytes;
    }

    static Params ToUnderlyingArguments(Arguments const& args, uint8_t* workspace)
    {
        GemmCoord problemShape = args.problemShape;
        uint32_t m = problemShape.m();
        uint32_t n = problemShape.n();
        uint32_t k = problemShape.k();
        LayoutA layoutA{m, k};
        LayoutB layoutB{k, n};

        uint8_t* evg_workspace = workspace + sizeof(ElementC) * static_cast<size_t>(m) * n;

        EpilogueParamsTuple epilogueParams = build_epilogue_params_tuple(problemShape, args.evg_args, evg_workspace, std::make_index_sequence<NumPhases>{});

        return Params{problemShape, args.ptrA, layoutA, args.ptrB, layoutB, workspace, epilogueParams};
    }

    CATLASS_DEVICE
    MatmulVisitorPhased() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const& params);

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const& params)
    {
        BlockScheduler matmulBlockScheduler(params.problemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        BlockMmad blockMmad(resource);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA*)params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB*)params.ptrB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC*)params.ptrWorkspace);
        layout::RowMajor layoutC(params.problemShape.m(), params.problemShape.n());

        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

            MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
            MatrixCoord offsetB{blockCoord.k() * L1TileShape::K, blockCoord.n() * L1TileShape::N};
            MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};
            int64_t gmOffsetA = params.layoutA.GetOffset(offsetA);
            int64_t gmOffsetB = params.layoutB.GetOffset(offsetB);
            int64_t gmOffsetC = layoutC.GetOffset(offsetC);

            blockMmad(
                gmA[gmOffsetA], params.layoutA,
                gmB[gmOffsetB], params.layoutB,
                gmC[gmOffsetC], layoutC,
                actualBlockShape);

            Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const& params)
    {
        // Common tensors on AIV side
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC*)params.ptrWorkspace);
        layout::RowMajor layoutC(params.problemShape.m(), params.problemShape.n());

        // Run each epilogue phase sequentially with a full-V barrier between phases
        run_all_phases(params, gmC, layoutC, std::make_index_sequence<NumPhases>{});

        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    template <size_t... Is>
    CATLASS_DEVICE void run_all_phases(
        Params const& params,
        AscendC::GlobalTensor<ElementC> const& gmC,
        layout::RowMajor const& layoutC,
        std::index_sequence<Is...>)
    {
        (run_one_phase<Is>(params, gmC, layoutC), ...);
    }

    template <size_t I>
    CATLASS_DEVICE void run_one_phase(
        Params const& params,
        AscendC::GlobalTensor<ElementC> const& gmC,
        layout::RowMajor const& layoutC)
    {
        using EpilogueT = std::tuple_element_t<I, EpilogueTuple>;
        EpilogueT blockEpilogue(resource, tla::get<I>(params.epilogueParams));

        BlockScheduler matmulBlockScheduler(params.problemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        GemmCoord blockShape = L1TileShape::ToCoord();

        uint32_t aicoreIndex = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t aicoreNum = AscendC::GetBlockNum();

        for (uint32_t loopIdx = aicoreIndex; loopIdx < coreLoops; loopIdx += aicoreNum) {
            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);
            auto gmBlockC = gmC[layoutC.GetOffset(blockCoord.GetCoordMN() * blockShape.GetCoordMN())];
            auto layoutBlockC = layoutC.GetTileLayout(actualBlockShape.GetCoordMN());

            if constexpr (I == 0) {
                // First phase waits for AIC store-complete per block
                Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagAicFinishStore);
            }

            blockEpilogue(blockShape, blockCoord, actualBlockShape, gmBlockC, layoutBlockC);
        }

        // Phase fence: ensure all V/MTE3 work is visible and aligned across all AIV cores
        AscendC::PipeBarrier<PIPE_ALL>();
        Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
    }

    template <size_t... Is>
    static bool can_implement_impl(Arguments const& args, std::index_sequence<Is...>)
    {
        bool ok = true;
        auto const& tup = args.evg_args;
        (void)std::initializer_list<int>{ (ok = ok && std::tuple_element_t<Is, EpilogueTuple>::EVG::can_implement(args.problemShape, std::get<Is>(tup)), 0)... };
        return ok;
    }

    template <size_t... Is>
    static size_t workspace_size_impl(Arguments const& args, std::index_sequence<Is...>)
    {
        size_t bytes = 0;
        auto const& tup = args.evg_args;
        (void)std::initializer_list<int>{ (bytes += std::tuple_element_t<Is, EpilogueTuple>::EVG::get_workspace_size(args.problemShape, std::get<Is>(tup)), 0)... };
        return bytes;
    }

    template <size_t... Is>
    static EpilogueParamsTuple build_epilogue_params_tuple(
        GemmCoord const& problemShape,
        typename MatmulVisitorPhased::EVGArgsTuple const& evg_args,
        uint8_t* evg_workspace_base,
        std::index_sequence<Is...>)
    {
        uint8_t* cur = evg_workspace_base;
        return tla::MakeTuple(
            build_one_param<Is>(problemShape, std::get<Is>(evg_args), cur)...
        );
    }

    template <size_t I>
    static auto build_one_param(
        GemmCoord const& problemShape,
        typename std::tuple_element_t<I, EpilogueTuple>::EVG::Arguments const& args,
        uint8_t*& cur)
    {
        using EVG = typename std::tuple_element_t<I, EpilogueTuple>::EVG;
        size_t need = EVG::get_workspace_size(problemShape, args);
        EVG::initialize_workspace(problemShape, args, cur);
        auto evg_params = EVG::to_underlying_arguments(problemShape, args, cur);
        cur += need;
        return typename std::tuple_element_t<I, EpilogueTuple>::Params{evg_params};
    }

private:
    static constexpr Arch::FlagID FLAG_AIC_FINISH_STORE = 0;
    static constexpr Arch::FlagID RV_FLAG_AIC_FINISH_STORE = 1;
    Arch::CrossCoreFlagWithReverse<> flagAicFinishStore{FLAG_AIC_FINISH_STORE, RV_FLAG_AIC_FINISH_STORE};
    Arch::Resource<ArchTag> resource;
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_MATMUL_VISITOR_PHASED_HPP


