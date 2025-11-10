#ifndef CATLASS_GEMM_KERNEL_MATMUL_MULTISTAGE_WORKSPACE_VISITOR_HPP
#define CATLASS_GEMM_KERNEL_MATMUL_MULTISTAGE_WORKSPACE_VISITOR_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/detail/callback.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

namespace Catlass::Gemm::Kernel {

template <
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_,
    uint32_t WORKSPACE_STAGES_
>
class MatmulMultiStageWorkspaceVisitor {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;

    using BlockEpilogue = BlockEpilogue_;
    using ElementD = typename BlockEpilogue::ElementD;
    using LayoutD = typename BlockEpilogue::LayoutD;
    using EpilogueParams = typename BlockEpilogue::Params;

    using BlockScheduler = BlockScheduler_;
    static constexpr uint32_t WORKSPACE_STAGES = WORKSPACE_STAGES_;

    /// Parameters structure
    struct Params {
        GemmCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrWorkspace;
        EpilogueParams epilogueParams;

        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(
            GemmCoord const& problemShape_,
            GM_ADDR ptrA_, LayoutA const& layoutA_,
            GM_ADDR ptrB_, LayoutB const& layoutB_,
            GM_ADDR ptrWorkspace_, EpilogueParams const& epilogueParams_
        ) : problemShape(problemShape_), ptrA(ptrA_), layoutA(layoutA_), ptrB(ptrB_), layoutB(layoutB_),
            ptrWorkspace(ptrWorkspace_), epilogueParams(epilogueParams_) {}
    };

    struct Arguments {
        GemmCoord problemShape;
        uint32_t aicCoreNum;
        GM_ADDR ptrA;
        GM_ADDR ptrB;
        typename BlockEpilogue::EVG::Arguments evg_args;
    };

    static bool CanImplement(const Arguments& args)
    {
        return BlockEpilogue::EVG::can_implement(args.problemShape, args.evg_args);
    }

    static size_t GetWorkspaceSize(const Arguments& args)
    {
        return static_cast<size_t>(L1TileShape::M) * L1TileShape::N *
        args.aicCoreNum * WORKSPACE_STAGES * sizeof(ElementC) +
               BlockEpilogue::EVG::get_workspace_size(args.problemShape, args.evg_args);
    }

    static Params ToUnderlyingArguments(const Arguments& args, uint8_t* workspace)
    {
        GemmCoord problemShape = args.problemShape;
        uint32_t m = problemShape.m();
        uint32_t n = problemShape.n();
        uint32_t k = problemShape.k();
        LayoutA layoutA{m, k};
        LayoutB layoutB{k, n};

        uint8_t* evg_workspace = workspace + static_cast<size_t>(L1TileShape::M) * L1TileShape::N *
        args.aicCoreNum * WORKSPACE_STAGES * sizeof(ElementC);
        BlockEpilogue::EVG::initialize_workspace(problemShape, args.evg_args, evg_workspace);

        // 转换 EVG Arguments 到 Params
        typename BlockEpilogue::EVG::Params fusion_params = 
            BlockEpilogue::EVG::to_underlying_arguments(
                problemShape, args.evg_args, 
                evg_workspace  // EVG workspace 在 GEMM workspace 之后
            );
        
        EpilogueParams epilogueParams{fusion_params};
        Params params{problemShape, args.ptrA, layoutA, args.ptrB, layoutB, workspace, epilogueParams};
        return params;
    }

    CATLASS_DEVICE
    MatmulMultiStageWorkspaceVisitor() {

        Arch::FlagID flagId = 0;
        for (uint32_t stageId = 0; stageId < WORKSPACE_STAGES; ++stageId) {
            flagAicFinishStoreList[stageId] = Arch::CrossCoreFlag(flagId++);
            flagAivFinishComputeList[stageId] = Arch::CrossCoreFlag(flagId++);
            aicWaitFuncList[stageId] = {this, stageId};
            aicSetFuncList[stageId] = {this, stageId};
        }
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const& params);

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const& params)
    {
        BlockScheduler blockScheduler;
        blockScheduler.Update(params.problemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
        uint32_t coreLoops = blockScheduler.GetCoreLoops();

        BlockMmad blockMmad(resource);

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA*)params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB*)params.ptrB);

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();

        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrWorkspace));
        auto layoutC = layout::RowMajor{L1TileShape::M * coreNum * WORKSPACE_STAGES, L1TileShape::N};

        uint32_t stageId = 0;
        uint32_t stageUsed = 0;

        // Loop through the matmul of each groupIdx
        for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNum) {
            // Compute block location
            GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);

            Callback callbackBeforeFixpipe{};
            if (stageUsed == WORKSPACE_STAGES) {
                callbackBeforeFixpipe = MakeCallback(&aicWaitFuncList[stageId]);
            } else {
                ++stageUsed;
            }
            Callback callbackAfterFixpipe = MakeCallback(&aicSetFuncList[stageId]);

            // Compute initial location in logical coordinates
            MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
            MatrixCoord offsetB{blockCoord.k() * L1TileShape::K, blockCoord.n() * L1TileShape::N};
            MatrixCoord offsetC{(stageId * coreNum + coreIdx) * L1TileShape::M, 0};
            int64_t gmOffsetA = params.layoutA.GetOffset(offsetA);
            int64_t gmOffsetB = params.layoutB.GetOffset(offsetB);
            int64_t gmOffsetC = layoutC.GetOffset(offsetC);

            // Compute block-scoped matrix multiply-add
            if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
                blockMmad(
                    gmA[gmOffsetA], params.layoutA,
                    gmB[gmOffsetB], params.layoutB,
                    gmC[gmOffsetC], layoutC,
                    actualBlockShape,
                    callbackBeforeFixpipe, callbackAfterFixpipe
                );
            } else {
                callbackBeforeFixpipe();
                blockMmad(
                    gmA[gmOffsetA], params.layoutA,
                    gmB[gmOffsetB], params.layoutB,
                    gmC[gmOffsetC], layoutC,
                    actualBlockShape
                );
                callbackAfterFixpipe();
            }

            stageId = (stageId + 1 < WORKSPACE_STAGES) ? (stageId + 1) : 0;
        }

        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
            blockMmad.SynchronizeBlock();
        }

        while (stageUsed > 0) {
            uint32_t aivComputeStageId = (stageId >= stageUsed) ?
                (stageId - stageUsed) : (stageId + WORKSPACE_STAGES - stageUsed);
            Arch::CrossCoreWaitFlag(flagAivFinishComputeList[aivComputeStageId]);
            --stageUsed;
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const& params)
    {
        BlockScheduler blockScheduler;
        BlockEpilogue blockEpilogue(resource, params.epilogueParams);

        uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t coreNum = AscendC::GetBlockNum();

        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrWorkspace));
        auto layoutC = layout::RowMajor{L1TileShape::M * coreNum * WORKSPACE_STAGES, L1TileShape::N};

        uint32_t stageId = 0;

        
        blockScheduler.Update(params.problemShape, L1TileShape::ToCoordMN());
        uint32_t coreLoops = blockScheduler.GetCoreLoops();

        GemmCoord blockShapeMNK = L1TileShape::ToCoord();
        for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNum) {
            GemmCoord blockCoordMNK = blockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShapeMNK = blockScheduler.GetActualBlockShape(blockCoordMNK);

            MatrixCoord offsetC{(stageId * coreNum + coreIdx) * L1TileShape::M, 0};
            int64_t gmOffsetC = layoutC.GetOffset(offsetC);
            auto gmBlockC = gmC[gmOffsetC];
            auto layoutBlockC = layoutC.GetTileLayout(actualBlockShapeMNK.GetCoordMN());

            Arch::CrossCoreWaitFlag(flagAicFinishStoreList[stageId]);
            blockEpilogue(blockShapeMNK, blockCoordMNK, actualBlockShapeMNK, gmBlockC, layoutBlockC);
            Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishComputeList[stageId]);

            stageId = (stageId + 1 < WORKSPACE_STAGES) ? (stageId + 1) : 0;
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
friend struct AicWaitFunc;
friend struct AicSetFunc;

struct AicWaitFunc {
    using MatmulKernel = MatmulMultiStageWorkspaceVisitor<BlockMmad, BlockEpilogue, BlockScheduler,
        WORKSPACE_STAGES>;

    CATLASS_DEVICE
    AicWaitFunc() = default;

    CATLASS_DEVICE
    void operator()() const
    {
        Arch::CrossCoreWaitFlag(ptr->flagAivFinishComputeList[stageId]);
    }

    MatmulKernel *ptr{nullptr};
    uint32_t stageId;
};

struct AicSetFunc {
    using MatmulKernel = MatmulMultiStageWorkspaceVisitor<BlockMmad, BlockEpilogue, BlockScheduler,
        WORKSPACE_STAGES>;

    CATLASS_DEVICE
    AicSetFunc() = default;

    CATLASS_DEVICE
    void operator()() const
    {
        Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(ptr->flagAicFinishStoreList[stageId]);
    }

    MatmulKernel *ptr{nullptr};
    uint32_t stageId;
};  

    Arch::CrossCoreFlag flagAicFinishStoreList[WORKSPACE_STAGES];
    Arch::CrossCoreFlag flagAivFinishComputeList[WORKSPACE_STAGES];

    AicWaitFunc aicWaitFuncList[WORKSPACE_STAGES];
    AicSetFunc aicSetFuncList[WORKSPACE_STAGES];
    Arch::Resource<ArchTag> resource;
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_MATMUL_MULTISTAGE_WORKSPACE_VISITOR_HPP



