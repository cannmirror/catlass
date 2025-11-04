#ifndef CATLASS_EPILOGUE_BLOCK_EPILOGUE_VISITOR_HPP
#define CATLASS_EPILOGUE_BLOCK_EPILOGUE_VISITOR_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/fusion/visitor_impl_base.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

namespace Catlass::Epilogue::Block {

template <
    class CType_,
    class ComputeLength_,
    class EVT_
>
class BlockEpilogue<
    EpilogueAtlasA2Visitor,
    CType_,
    ComputeLength_,
    EVT_
> {
public:
    using DispatchPolicy = EpilogueAtlasA2Visitor;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using ElementD = ElementC;
    using LayoutD = LayoutC;

    static constexpr uint32_t COMPUTE_LENGTH = ComputeLength_::value;
    using EVT = EVT_;

    struct Params {
        typename EVT::Params evt_params;

        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(typename EVT::Params const& evt_params_)
            : evt_params(evt_params_) {}
    };

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag>& resource, Params const& params)
        : params(params), evt(params.evt_params), resource_(resource)
    {
        uint32_t ub_offset0 = 0;
        auto callbacks0 = evt.get_callbacks(
            resource_, ub_offset0, COMPUTE_LENGTH
        );
        callbacks0.begin_epilogue();
        // 事件ID分配：四类事件分别独立编号（每类两路 buffer）
        int32_t evVMTE2 = 0;   // V_MTE2
        int32_t evMTE2V = 0;   // MTE2_V
        int32_t evMTE3V = 0;   // MTE3_V
        int32_t evVMTE3 = 0;   // V_MTE3

        for (int i = 0; i < 2; ++i) {
            eventVMTE2[i] = evVMTE2++;
            eventMTE2V[i] = evMTE2V++;
            eventMTE3V[i] = evMTE3V++;
            eventVMTE3[i] = evVMTE3++;
        }

        // 初始状态：允许搬入（V_MTE2 已完成）与允许写入（MTE3_V 已完成）
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventVMTE2[0]);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventVMTE2[1]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventMTE3V[0]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventMTE3V[1]);
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventVMTE2[0]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventVMTE2[1]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventMTE3V[0]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventMTE3V[1]);

        // AscendC::PipeBarrier<PIPE_ALL>();
        // Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();

        uint32_t ub_offset0 = 0;
        auto callbacks0 = evt.get_callbacks(
            resource_, ub_offset0, COMPUTE_LENGTH
        );
        callbacks0.end_epilogue();
    }

    CATLASS_DEVICE
    void operator()(
        GemmCoord const& blockShapeMNK,
        GemmCoord const& blockCoordMNK,
        GemmCoord const& actualBlockShapeMNK,
        AscendC::GlobalTensor<ElementC> const& gmBlockC,
        layout::RowMajor const& layoutBlockC
    )
    {
        MatrixCoord blockShape = blockShapeMNK.GetCoordMN();
        MatrixCoord blockCoord = blockCoordMNK.GetCoordMN();
        MatrixCoord actualBlockShape = actualBlockShapeMNK.GetCoordMN();
        MatrixCoord blockOffset = blockCoord * blockShape;

        // 子块划分
        MatrixCoord subblockShape{
            CeilDiv(actualBlockShape.row(), static_cast<uint32_t>(AscendC::GetSubBlockNum())),
            actualBlockShape.column()
        };
        MatrixCoord subblockCoord{AscendC::GetSubBlockIdx(), 0};
        MatrixCoord actualSubblockShape = MatrixCoord::Min(
            subblockShape, actualBlockShape - subblockCoord * subblockShape);
        MatrixCoord subblockOffset = subblockCoord * subblockShape;

        // 获取 gmSubblockC 和 layoutSubblockC
        auto gmSubblockC = gmBlockC[layoutBlockC.GetOffset(subblockOffset)];
        auto layoutSubblockC = layoutBlockC.GetTileLayout(actualSubblockShape);

        // 分配 UB 空间并获取两套 callbacks（双缓冲）
        uint32_t ub_offset0 = 0;
        auto callbacks0 = evt.get_callbacks(
            resource_, ub_offset0, COMPUTE_LENGTH
        );
        uint32_t ub_offset1 = ub_offset0;
        auto callbacks1 = evt.get_callbacks(
            resource_, ub_offset1, COMPUTE_LENGTH
        );

        uint32_t rows = actualSubblockShape.row();
        uint32_t cols = actualSubblockShape.column();

        // 遍历所有 tile，实现双缓冲流水
        uint32_t ubListId = 0;  // 0或1，交替使用
        
        for (uint32_t r = 0; r < rows; ) {
            auto& cbs = ((ubListId & 1) ? callbacks1 : callbacks0);

            // 检查是否需要列分块
            if (cols <= COMPUTE_LENGTH) {
                // 列宽 <= COMPUTE_LENGTH，可以处理完整行宽
                uint32_t colsAligned = RoundUp<BYTE_PER_C0>(cols);
                uint32_t maxRowsPerTile = COMPUTE_LENGTH / colsAligned;
                if (maxRowsPerTile == 0) maxRowsPerTile = 1;  // 防止除零
                
                uint32_t remainRows = rows - r;
                uint32_t tileRows = (remainRows < maxRowsPerTile) ? remainRows : maxRowsPerTile;
                
                MatrixCoord tileShape{tileRows, cols};
                MatrixCoord localTileOffset{r, 0};
                // 计算全局绝对坐标
                MatrixCoord globalTileOffset = blockOffset + subblockOffset + localTileOffset;
                uint32_t calCount = tileRows * colsAligned;
                
                // 计算对齐的 tile shape
                MatrixCoord alignedTileShape{
                    tileShape.row(),
                    colsAligned
                };
                
                // 统一流水：执行一次 tile 的 Load-Compute-Store
                run_tile(cbs, gmSubblockC, layoutSubblockC, globalTileOffset, localTileOffset, tileShape, alignedTileShape, calCount, ubListId);
                r += tileRows;
            } else { //应该暂时都用不到
                // 列宽 > COMPUTE_LENGTH，需要列分块，每次处理1行
                for (uint32_t c = 0; c < cols; ) {
                    uint32_t remainCols = cols - c;
                    uint32_t tileCols = (remainCols < COMPUTE_LENGTH) ? remainCols : COMPUTE_LENGTH;
                    
                    uint32_t colsAligned = RoundUp<BYTE_PER_C0>(tileCols);

                    MatrixCoord tileShape{1, tileCols};
                    MatrixCoord localTileOffset{r, c};
                    // 计算全局绝对坐标
                    MatrixCoord globalTileOffset = blockOffset + subblockOffset + localTileOffset;
                    uint32_t calCount = tileCols;  // 1行 * tileCols列
                    
                    // 计算对齐的 tile shape
                    MatrixCoord alignedTileShape{
                        tileShape.row(),
                        colsAligned
                    };
                    
                    // 统一流水：执行一次 tile 的 Load-Compute-Store
                    run_tile(cbs, gmSubblockC, layoutSubblockC, globalTileOffset, localTileOffset, tileShape, alignedTileShape, calCount, ubListId);
                    c += tileCols;
                }
                
                r += 1;  // 处理完一行
            }

            ubListId = 1 - ubListId; // Buffer 轮转
        }
    }

private:
    template <class Callbacks>
    CATLASS_DEVICE void run_tile(
        Callbacks& cbs,
        AscendC::GlobalTensor<ElementC> const& gmSubblockC,
        layout::RowMajor const& layoutSubblockC,
        MatrixCoord const& globalTileOffset,
        MatrixCoord const& localTileOffset,
        MatrixCoord const& actualTileShape,
        MatrixCoord const& alignedTileShape,
        uint32_t calCount,
        uint32_t ubListId
    ) {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventVMTE2[ubListId]);
        cbs.visit(gmSubblockC, layoutSubblockC, globalTileOffset, localTileOffset, actualTileShape, alignedTileShape, calCount, Epilogue::Fusion::VisitStage::LOAD);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventMTE2V[ubListId]);

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventMTE2V[ubListId]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventMTE3V[ubListId]);
        cbs.visit(gmSubblockC, layoutSubblockC, globalTileOffset, localTileOffset, actualTileShape, alignedTileShape, calCount, Epilogue::Fusion::VisitStage::COMPUTE);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventVMTE2[ubListId]);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventVMTE3[ubListId]);

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventVMTE3[ubListId]);
        cbs.visit(gmSubblockC, layoutSubblockC, globalTileOffset, localTileOffset, actualTileShape, alignedTileShape, calCount, Epilogue::Fusion::VisitStage::STORE);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventMTE3V[ubListId]);
    }

    Params params;
    EVT evt;
    Arch::Resource<ArchTag>& resource_;  // 新增成员引用
    int32_t eventVMTE2[2];   // V_MTE2：V->MTE2，同步事件（两个buffer）
    int32_t eventMTE2V[2];   // MTE2_V：MTE2->V，同步事件（两个buffer）
    int32_t eventMTE3V[2];   // MTE3_V：MTE3->V，同步事件（两个buffer）
    int32_t eventVMTE3[2];   // V_MTE3：V->MTE3，同步事件（两个buffer）
};

} // namespace Catlass::Epilogue::Block

#endif // CATLASS_EPILOGUE_BLOCK_EPILOGUE_VISITOR_HPP
