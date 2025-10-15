#ifndef CATLASS_EPILOGUE_FUSION_VISITOR_ROW_REDUCE_HPP
#define CATLASS_EPILOGUE_FUSION_VISITOR_ROW_REDUCE_HPP

#include "catlass/epilogue/fusion/visitor_impl.hpp"
#include "catlass/epilogue/fusion/operations.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"

namespace Catlass::Epilogue::Fusion {

// Row reduction over the tile along rows: sum each column of the MxN tile into a 1xN vector,
// then atomically add to a GM row vector. Returns input tensor to allow chaining.
template <class ElementCompute, class Layout = layout::RowMajor>
struct VisitorRowReduce : VisitorImpl<> {
    using VisitorImpl<>::VisitorImpl;

    // For TopologicalVisitor cache and composition, expose input element type
    using ElementOutput = ElementCompute;
    static constexpr VisitStage OUTPUT_STAGE = VisitStage::STORE;

    struct Arguments {
        GM_ADDR ptr_row_out = nullptr; // GM address of 1 x N output row vector
        Layout layout = {};            // layout over (1, N)
        ElementCompute identity = ElementCompute(0);
    };

    struct Params {
        GM_ADDR ptr_row_out;
        Layout layout;
        ElementCompute identity;

        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GM_ADDR ptr_row_out_, Layout const& layout_, ElementCompute const& identity_)
            : ptr_row_out(ptr_row_out_), layout(layout_), identity(identity_) {}
    };

    template <class ProblemShape>
    CATLASS_HOST_DEVICE static constexpr Params
    to_underlying_arguments(ProblemShape const&, Arguments const& args, void*) {
        return Params(args.ptr_row_out, args.layout, args.identity);
    }

    template <class ProblemShape>
    CATLASS_HOST_DEVICE static size_t
    get_workspace_size(ProblemShape const&, Arguments const&) {
        return 0;
    }

    template <class ProblemShape>
    CATLASS_HOST_DEVICE static bool
    can_implement(ProblemShape const&, Arguments const& args) {
        return args.ptr_row_out != nullptr;
    }

    CATLASS_HOST_DEVICE
    VisitorRowReduce() {}

    CATLASS_HOST_DEVICE
    VisitorRowReduce(Params const& params_) : params(params_) {}

    struct Callbacks : EmptyCallbacks {
        AscendC::LocalTensor<ElementCompute> ubReduce;   // accumulation buffer (length >= cols)
        Params const* params_ptr;
        uint32_t compute_length;

        CATLASS_DEVICE
        Callbacks(
            AscendC::LocalTensor<ElementCompute> ubReduce_,
            Params const* params_ptr_,
            uint32_t compute_length_)
            : ubReduce(ubReduce_), params_ptr(params_ptr_), compute_length(compute_length_) {}


        template <typename ElementInput>
        CATLASS_DEVICE AscendC::LocalTensor<ElementInput> const& visit(
            MatrixCoord const& globalTileOffset,
            MatrixCoord const& /*localTileOffset*/,
            MatrixCoord const& tileShape,      // (rows, cols)
            uint32_t /*calCount*/,
            VisitStage stage,
            AscendC::LocalTensor<ElementInput> const& input)
        {
            uint32_t rows = tileShape.row();
            uint32_t cols = tileShape.column();

            if (stage == VisitStage::COMPUTE || stage == VisitStage::ALL) {
                // Require exact type match; no implicit cast here
                static_assert(std::is_same_v<ElementInput, ElementCompute>,
                              "VisitorRowReduce expects input type == ElementCompute. Insert VisitorCast if needed.");
                // Reduce along rows into ubReduce[0:cols)

                AscendC::Duplicate(ubReduce, params_ptr->identity, compute_length);
                if (rows > 0) {
                    AscendC::DataCopy(ubReduce, input, cols);
                }
                for (uint32_t r = 1; r < rows; ++r) {
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(ubReduce, ubReduce, input[r * cols], cols);
                }
            }

            if (stage == VisitStage::STORE || stage == VisitStage::ALL) {
                // Atomic add ubReduce[0:cols) into GM row segment
                AscendC::GlobalTensor<ElementCompute> gmRowOut;
                gmRowOut.SetGlobalBuffer((__gm__ ElementCompute*)(params_ptr->ptr_row_out));
                auto gmTile = gmRowOut[params_ptr->layout.GetOffset(MatrixCoord{0, globalTileOffset.column()})];

                // Prepare UB view for 1 x cols
                auto layoutUbRowOut = layout::RowMajor::MakeLayoutInUb<ElementCompute>(MatrixCoord{1u, cols});
                using CopyUb2GmT = Epilogue::Tile::CopyUb2Gm<Arch::AtlasA2, Gemm::GemmType<ElementCompute, layout::RowMajor>>;
                CopyUb2GmT copyUb2Gm{};
                auto layoutDst = params_ptr->layout.GetTileLayout(MatrixCoord{1u, cols});
                AscendC::SetAtomicAdd<ElementCompute>();
                copyUb2Gm(gmTile, ubReduce, layoutDst, layoutUbRowOut);
                AscendC::SetAtomicNone();
            }
            return input;
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
        auto ubReduce = resource.ubBuf.template GetBufferByByte<ElementCompute>(ub_offset);
        ub_offset += compute_length * sizeof(ElementCompute);
        return Callbacks(ubReduce, &params, compute_length);
    }

    Params params;
};

} // namespace Catlass::Epilogue::Fusion

#endif


