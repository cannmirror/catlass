#ifndef CATLASS_EPILOGUE_FUSION_VISITOR_COL_BROADCAST_HPP
#define CATLASS_EPILOGUE_FUSION_VISITOR_COL_BROADCAST_HPP

#include "catlass/epilogue/fusion/visitor_impl.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"

namespace Catlass::Epilogue::Fusion {

// Column-broadcast an Mx1 GM vector to an MxN UB tile for the current tile.
// LOAD stage: load GM column segment [globalTileOffset.row() : rows, 0] -> UB (rows x 1),
// COMPUTE stage: replicate across columns into an MxN contiguous UB buffer and return it.
template <class Element, class Layout = layout::RowMajor>
struct VisitorColBroadcast : VisitorImpl<> {
    using VisitorImpl<>::VisitorImpl;

    using ElementOutput = Element;
    static constexpr VisitStage OUTPUT_STAGE = VisitStage::COMPUTE;

    struct Arguments {
        GM_ADDR ptr_col = nullptr;  // GM address of M x 1 column vector
        Layout layout = {};          // layout over (M, 1)
    };

    struct Params {
        GM_ADDR ptr_col;
        Layout layout;

        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GM_ADDR ptr_col_, Layout const& layout_)
            : ptr_col(ptr_col_), layout(layout_) {}
    };

    template <class ProblemShape>
    CATLASS_HOST_DEVICE static constexpr Params
    to_underlying_arguments(ProblemShape const&, Arguments const& args, void*) {
        return Params(args.ptr_col, args.layout);
    }

    template <class ProblemShape>
    CATLASS_HOST_DEVICE static size_t
    get_workspace_size(ProblemShape const&, Arguments const&) {
        return 0;
    }

    template <class ProblemShape>
    CATLASS_HOST_DEVICE static bool
    can_implement(ProblemShape const&, Arguments const& args) {
        return args.ptr_col != nullptr;
    }

    CATLASS_HOST_DEVICE
    VisitorColBroadcast() {}

    CATLASS_HOST_DEVICE
    VisitorColBroadcast(Params const& params_) : params(params_) {}

    struct Callbacks : EmptyCallbacks {
        AscendC::LocalTensor<Element> ubOut;  // size >= compute_length
        Params const* params_ptr;
        uint32_t compute_length;

        CATLASS_DEVICE
        Callbacks(
            AscendC::LocalTensor<Element> ubOut_,
            Params const* params_ptr_,
            uint32_t compute_length_)
            : ubOut(ubOut_), params_ptr(params_ptr_), compute_length(compute_length_) {}

        template <typename ElementAccumulator, typename... Args>
        CATLASS_DEVICE AscendC::LocalTensor<Element> const& visit(
            AscendC::GlobalTensor<ElementAccumulator> const& /*gmSubblockC*/,
            layout::RowMajor const& /*layoutSubblockC*/,
            MatrixCoord const& globalTileOffset,
            MatrixCoord const& /*localTileOffset*/,    // not needed here
            MatrixCoord const& actualTileShape,
            MatrixCoord const& alignedTileShape,
            uint32_t /*calCount*/,
            VisitStage stage,
            Args const&... /*unused*/)
        {
            uint32_t actualRows = actualTileShape.row();
            uint32_t actualCols = actualTileShape.column();
            uint32_t alignedCols = alignedTileShape.column();

            if (stage == VisitStage::LOAD) {
                // GM load 使用 actualTileShape 的行数，列为 1
                auto layoutUbCol = layout::RowMajor{1, actualRows};
                using CopyGm2UbT = Epilogue::Tile::CopyGm2Ub<Arch::AtlasA2, Gemm::GemmType<Element, layout::RowMajor>>;
                CopyGm2UbT copyGm2Ub{};

                AscendC::GlobalTensor<Element> gmCol;
                gmCol.SetGlobalBuffer((__gm__ Element*)(params_ptr->ptr_col));
                auto gmTile = gmCol[params_ptr->layout.GetOffset(MatrixCoord{globalTileOffset.row(), 0})];
                auto layoutSrc = params_ptr->layout.GetTileLayout(MatrixCoord{1, actualRows});
                copyGm2Ub(ubOut, gmTile, layoutUbCol, layoutSrc); // writes first column of ubOut per row
            }
            if (stage == VisitStage::COMPUTE) {
                // 列向广播：LOAD阶段将每行标量按{actualRows,1}布局写入ubOut的连续位置r
                // 此处从偏移r读取，然后复制到按alignedCols对齐的一整行
                AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
                for (int r = actualRows-1; r >= 0; --r) {
                    Element v = ubOut.GetValue(r);
                    AscendC::Duplicate<Element>(ubOut[r * alignedCols], v, actualCols);
                }
            }
            return ubOut;
        }
    };

    template <class ArchTag>
    CATLASS_DEVICE auto get_callbacks(
        Arch::Resource<ArchTag>& resource,
        uint32_t& ub_offset,
        uint32_t compute_length)
    {
        auto ubOut = resource.ubBuf.template GetBufferByByte<Element>(ub_offset);
        ub_offset += compute_length * sizeof(Element);
        assert(ub_offset <= ArchTag::UB_SIZE, "ub_offset exceeds ArchTag::UB_SIZE");
        return Callbacks(ubOut, &params, compute_length);
    }

    Params params;
};

} // namespace Catlass::Epilogue::Fusion

#endif


