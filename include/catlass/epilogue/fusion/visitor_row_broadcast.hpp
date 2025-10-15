#ifndef CATLASS_EPILOGUE_FUSION_VISITOR_ROW_BROADCAST_HPP
#define CATLASS_EPILOGUE_FUSION_VISITOR_ROW_BROADCAST_HPP

#include "catlass/epilogue/fusion/visitor_impl.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"

namespace Catlass::Epilogue::Fusion {

// Row-broadcast a 1xN GM vector to an MxN UB tile for the current tile.
// LOAD stage: load GM row segment [0, globalTileOffset.column() : cols] -> UB (1 x cols),
//             then replicate across rows into an MxN contiguous UB buffer and return it.
// COMPUTE/STORE stage: no-op, just return the cached UB buffer.
template <class Element, class Layout = layout::RowMajor>
struct VisitorRowBroadcast : VisitorImpl<> {
    using VisitorImpl<>::VisitorImpl;

    using ElementOutput = Element;
    static constexpr VisitStage OUTPUT_STAGE = VisitStage::LOAD;

    struct Arguments {
        GM_ADDR ptr_row = nullptr;  // GM address of 1 x N row vector
        Layout layout = {};          // layout over (1, N)
    };

    struct Params {
        GM_ADDR ptr_row;
        Layout layout;

        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GM_ADDR ptr_row_, Layout const& layout_)
            : ptr_row(ptr_row_), layout(layout_) {}
    };

    template <class ProblemShape>
    CATLASS_HOST_DEVICE static constexpr Params
    to_underlying_arguments(ProblemShape const&, Arguments const& args, void*) {
        return Params(args.ptr_row, args.layout);
    }

    template <class ProblemShape>
    CATLASS_HOST_DEVICE static size_t
    get_workspace_size(ProblemShape const&, Arguments const&) {
        return 0;
    }

    template <class ProblemShape>
    CATLASS_HOST_DEVICE static bool
    can_implement(ProblemShape const&, Arguments const& args) {
        return args.ptr_row != nullptr;
    }

    CATLASS_HOST_DEVICE
    VisitorRowBroadcast() {}

    CATLASS_HOST_DEVICE
    VisitorRowBroadcast(Params const& params_) : params(params_) {}

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

        template <typename... Args>
        CATLASS_DEVICE AscendC::LocalTensor<Element> const& visit(
            MatrixCoord const& globalTileOffset,
            MatrixCoord const& /*localTileOffset*/,    // not needed here
            MatrixCoord const& tileShape,              // (rows, cols)
            uint32_t /*calCount*/,
            VisitStage stage,
            Args const&... /*unused*/)
        {
            uint32_t rows = tileShape.row();
            uint32_t cols = tileShape.column();
            if (stage == VisitStage::LOAD || stage == VisitStage::ALL) {
                // Load GM row segment [0, colOffset : colOffset+cols) into the first row of ubOut
                auto layoutUbRow = layout::RowMajor::MakeLayoutInUb<Element>(MatrixCoord{1u, cols});
                using CopyGm2UbT = Epilogue::Tile::CopyGm2Ub<Arch::AtlasA2, Gemm::GemmType<Element, layout::RowMajor>>;
                CopyGm2UbT copyGm2Ub{};

                AscendC::GlobalTensor<Element> gmRow;
                gmRow.SetGlobalBuffer((__gm__ Element*)(params_ptr->ptr_row));
                auto gmTile = gmRow[params_ptr->layout.GetOffset(MatrixCoord{0, globalTileOffset.column()})];
                auto layoutSrc = params_ptr->layout.GetTileLayout(MatrixCoord{1u, cols});
                copyGm2Ub(ubOut, gmTile, layoutUbRow, layoutSrc); // writes ubOut[0..cols-1]
            }
            if (stage == VisitStage::COMPUTE || stage == VisitStage::ALL) {
                // Replicate the first row across remaining rows into ubOut as an MxN tile
                if (rows > 1) {
                    for (uint32_t r = 1; r < rows; ++r) {
                        // copy row 0 -> row r
                        AscendC::DataCopy(
                            ubOut[r * cols],   // dst start at offset r*cols
                            ubOut[0],          // src start at offset 0
                            cols);
                    }
                }
            }
            return ubOut;
        }
    };

    template <class ArchTag>
    CATLASS_DEVICE auto get_callbacks(
        Arch::Resource<ArchTag>& resource,
        uint32_t& ub_offset,
        uint32_t compute_length,
        GemmCoord const&,
        GemmCoord const&,
        MatrixCoord const&,
        MatrixCoord const&,
        AscendC::GlobalTensor<half> const&,
        layout::RowMajor const&)
    {
        auto ubOut = resource.ubBuf.template GetBufferByByte<Element>(ub_offset);
        ub_offset += compute_length * sizeof(Element);
        return Callbacks(ubOut, &params, compute_length);
    }

    Params params;
};

} // namespace Catlass::Epilogue::Fusion

#endif


