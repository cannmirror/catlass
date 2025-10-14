#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_VISITOR_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_VISITOR_HPP

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
    class ElementCompute_,
    class FusionCallbacks_
>
class BlockEpilogue<
    EpilogueWithVisitorCallbacks,
    CType_,
    ComputeLength_,
    ElementCompute_,
    FusionCallbacks_
> {
public:
    using DispatchPolicy = EpilogueWithVisitorCallbacks;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using ElementD = ElementC;
    using LayoutD = LayoutC;

    static constexpr uint32_t COMPUTE_LENGTH = ComputeLength_::value;
    using ElementCompute = ElementCompute_;
    using FusionCallbacks = FusionCallbacks_;

    struct Params {
        typename FusionCallbacks::Params fusion_params;

        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(typename FusionCallbacks::Params const& fusion_params_)
            : fusion_params(fusion_params_) {}
    };

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag>& resource, Params const& params)
        : params(params), fusion_callbacks(params.fusion_params)
    {
        // 为两个buffer分配独立的event_id
        eventV_MTE2[0] = 0;
        eventV_MTE2[1] = 1;
        eventMTE3_V[0] = 2;
        eventMTE3_V[1] = 3;
        
        // 初始状态：允许搬入和搬出
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventV_MTE2[0]);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventV_MTE2[1]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventMTE3_V[0]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventMTE3_V[1]);
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventV_MTE2[0]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventV_MTE2[1]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventMTE3_V[0]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventMTE3_V[1]);
    }

    CATLASS_DEVICE
    void operator()(
        Arch::Resource<ArchTag>& resource,
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
        auto callbacks0 = fusion_callbacks.get_callbacks(
            resource, ub_offset0, COMPUTE_LENGTH,
            blockShapeMNK, blockCoordMNK,
            actualSubblockShape, subblockCoord,
            gmSubblockC, layoutSubblockC
        );
        uint32_t ub_offset1 = ub_offset0;
        auto callbacks1 = fusion_callbacks.get_callbacks(
            resource, ub_offset1, COMPUTE_LENGTH,
            blockShapeMNK, blockCoordMNK,
            actualSubblockShape, subblockCoord,
            gmSubblockC, layoutSubblockC
        );

        callbacks0.begin_epilogue();
        callbacks1.begin_epilogue();

        uint32_t rows = actualSubblockShape.row();
        uint32_t cols = actualSubblockShape.column();

        // 遍历所有 tile，实现双缓冲流水
        uint32_t tileIdx = 0;
        uint32_t ubListId = 0;  // 0或1，交替使用
        
        for (uint32_t r = 0; r < rows; ) {
            auto& cbsRow = ((ubListId & 1) ? callbacks1 : callbacks0);
            cbsRow.begin_row(r);

            // 检查是否需要列分块
            if (cols <= COMPUTE_LENGTH) {
                // 列宽 <= COMPUTE_LENGTH，可以处理完整行宽
                uint32_t maxRowsPerTile = COMPUTE_LENGTH / cols;
                if (maxRowsPerTile == 0) maxRowsPerTile = 1;  // 防止除零
                
                uint32_t remainRows = rows - r;
                uint32_t tileRows = (remainRows < maxRowsPerTile) ? remainRows : maxRowsPerTile;
                
                MatrixCoord tileShape{tileRows, cols};
                MatrixCoord localTileOffset{r, 0};
                // 计算全局绝对坐标
                MatrixCoord globalTileOffset = blockOffset + subblockOffset + localTileOffset;
                uint32_t calCount = tileRows * cols;

                // === Load阶段：等待上一次compute完成 ===
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventV_MTE2[ubListId]);
                
                // 调用EVT树load阶段（MTE2流水）
                auto& cbs = ((ubListId & 1) ? callbacks1 : callbacks0);
                cbs.visit(globalTileOffset, localTileOffset, tileShape, calCount, 
                          Epilogue::Fusion::VisitStage::LOAD);
                
                // Load完成，通知compute可以开始
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventV_MTE2[ubListId]);
                
                // === Compute阶段：等待load完成 & store空闲 ===
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventV_MTE2[ubListId]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventMTE3_V[ubListId]);
                
                // 调用EVT树compute阶段（V流水）
                cbs.visit(globalTileOffset, localTileOffset, tileShape, calCount,
                          Epilogue::Fusion::VisitStage::COMPUTE);
                
                // Compute完成，通知load可以覆盖 & store可以搬出
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventV_MTE2[ubListId]);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventMTE3_V[ubListId]);
                
                // === Store阶段：等待compute完成 ===
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventMTE3_V[ubListId]);
                
                // 调用EVT树store阶段（MTE3流水）
                cbs.visit(globalTileOffset, localTileOffset, tileShape, calCount,
                          Epilogue::Fusion::VisitStage::STORE);
                
                // Store完成，通知compute可以写入
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventMTE3_V[ubListId]);
                
                // Buffer轮转
                ubListId = 1 - ubListId;
                ++tileIdx;
                r += tileRows;
            } else { //应该暂时都用不到
                // 列宽 > COMPUTE_LENGTH，需要列分块，每次处理1行
                for (uint32_t c = 0; c < cols; ) {
                    uint32_t remainCols = cols - c;
                    uint32_t tileCols = (remainCols < COMPUTE_LENGTH) ? remainCols : COMPUTE_LENGTH;
                    
                    MatrixCoord tileShape{1, tileCols};
                    MatrixCoord localTileOffset{r, c};
                    // 计算全局绝对坐标
                    MatrixCoord globalTileOffset = blockOffset + subblockOffset + localTileOffset;
                    uint32_t calCount = tileCols;  // 1行 * tileCols列

                    // === Load阶段：等待上一次compute完成 ===
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventV_MTE2[ubListId]);
                    
                    // 调用EVT树load阶段（MTE2流水）
                    auto& cbs = ((ubListId & 1) ? callbacks1 : callbacks0);
                    cbs.visit(globalTileOffset, localTileOffset, tileShape, calCount, 
                              Epilogue::Fusion::VisitStage::LOAD);
                    
                    // Load完成，通知compute可以开始
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventV_MTE2[ubListId]);
                    
                    // === Compute阶段：等待load完成 & store空闲 ===
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventV_MTE2[ubListId]);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventMTE3_V[ubListId]);
                    
                    // 调用EVT树compute阶段（V流水）
                    cbs.visit(globalTileOffset, localTileOffset, tileShape, calCount,
                              Epilogue::Fusion::VisitStage::COMPUTE);
                    
                    // Compute完成，通知load可以覆盖 & store可以搬出
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventV_MTE2[ubListId]);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventMTE3_V[ubListId]);
                    
                    // === Store阶段：等待compute完成 ===
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventMTE3_V[ubListId]);
                    
                    // 调用EVT树store阶段（MTE3流水）
                    cbs.visit(globalTileOffset, localTileOffset, tileShape, calCount,
                              Epilogue::Fusion::VisitStage::STORE);
                    
                    // Store完成，通知compute可以写入
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventMTE3_V[ubListId]);
                    
                    // Buffer轮转
                    ubListId = 1 - ubListId;
                    ++tileIdx;
                    c += tileCols;
                }
                
                r += 1;  // 处理完一行
            }

            auto& cbsEnd = ((ubListId & 1) ? callbacks1 : callbacks0);
            cbsEnd.end_row(r);
        }

        callbacks0.end_epilogue();
        callbacks1.end_epilogue();
    }

private:
    Params params;
    FusionCallbacks fusion_callbacks;
    int32_t eventV_MTE2[2];  // MTE2→V同步事件（两个buffer）
    int32_t eventMTE3_V[2];  // V→MTE3同步事件（两个buffer）
};

} // namespace Catlass::Epilogue::Block

#endif
