#ifndef CATLASS_EPILOGUE_FUSION_VISITOR_ROW_REDUCE_HPP
#define CATLASS_EPILOGUE_FUSION_VISITOR_ROW_REDUCE_HPP

#include "catlass/epilogue/fusion/visitor_impl.hpp"
#include "catlass/epilogue/fusion/operations.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"

namespace Catlass::Epilogue::Fusion {

// Row reduction over the tile along rows: reduce each column of the MxN tile into a 1xN vector,
// then atomically reduce to a GM row vector. Returns input tensor to allow chaining.
template <template<class> class ReduceFn, class Element, class Layout = layout::RowMajor, ReduceStrategy Strategy = ReduceStrategy::ATOMIC_REDUCE>
struct VisitorRowReduce : VisitorImpl<> {
    using VisitorImpl<>::VisitorImpl;

    // For TopologicalVisitor cache and composition, expose input element type
    using ElementOutput = Element;
    static constexpr VisitStage OUTPUT_STAGE = VisitStage::STORE;

    struct Arguments {
        GM_ADDR ptr_row_out = nullptr; // GM address of 1 x N output row vector
        Layout layout = {};            // layout over (1, N)
        Element identity = Element(0);
    };

    struct Params {
        GM_ADDR ptr_row_out;
        Layout layout;
        Element identity;

        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GM_ADDR ptr_row_out_, Layout const& layout_, Element const& identity_)
            : ptr_row_out(ptr_row_out_), layout(layout_), identity(identity_) {}
    };

    template <class ProblemShape>
    CATLASS_HOST_DEVICE static constexpr Params
    to_underlying_arguments(ProblemShape const&, Arguments const& args, void*) {
        return Params(args.ptr_row_out, args.layout, args.identity);
    }

    template <class ProblemShape>
    CATLASS_HOST_DEVICE static size_t
    get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
        if constexpr (Strategy == ReduceStrategy::WORKSPACE_REDUCE) {
            // TODO: 计算 workspace 大小
            return 0;
        }
        return 0;
    }

    template <class ProblemShape>
    CATLASS_HOST_DEVICE static bool
    initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
        if constexpr (Strategy == ReduceStrategy::WORKSPACE_REDUCE) {
            // TODO: 初始化 workspace
            return true;
        }
        return true;
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
        AscendC::LocalTensor<Element> ubReduce;   // accumulation buffer (length >= cols)
        Params const* params_ptr;
        uint32_t compute_length;

        CATLASS_DEVICE
        Callbacks(
            AscendC::LocalTensor<Element> ubReduce_,
            Params const* params_ptr_,
            uint32_t compute_length_)
            : ubReduce(ubReduce_), params_ptr(params_ptr_), compute_length(compute_length_) {}


        template <typename ElementAccumulator, typename ElementInput>
        CATLASS_DEVICE AscendC::LocalTensor<ElementInput> const& visit(
            AscendC::GlobalTensor<ElementAccumulator> const& /*gmSubblockC*/,
            layout::RowMajor const& /*layoutSubblockC*/,
            MatrixCoord const& globalTileOffset,
            MatrixCoord const& /*localTileOffset*/,
            MatrixCoord const& actualTileShape,
            MatrixCoord const& alignedTileShape,
            uint32_t /*calCount*/,
            VisitStage stage,
            AscendC::LocalTensor<ElementInput> const& input)
        {
            uint32_t actualRows = actualTileShape.row();
            uint32_t actualCols = actualTileShape.column();
            uint32_t alignedCols = alignedTileShape.column();

            if (stage == VisitStage::COMPUTE) {
                // Require exact type match; no implicit cast here
                static_assert(std::is_same_v<ElementInput, Element>,
                              "VisitorRowReduce expects input type == Element. Insert VisitorCast if needed.");
                
                // 使用策略化的计算逻辑
                RowReduceCompute<ReduceFn, Element, Strategy>::compute(
                    ubReduce, input, params_ptr->identity, actualRows, actualCols, alignedCols);
            }

            if (stage == VisitStage::STORE) {
                // 使用策略化的存储逻辑
                AscendC::GlobalTensor<Element> gmRowOut;
                gmRowOut.SetGlobalBuffer((__gm__ Element*)(params_ptr->ptr_row_out));
                
                RowReduceCompute<ReduceFn, Element, Strategy>::store(
                    gmRowOut, ubReduce, globalTileOffset, actualTileShape, params_ptr->layout);
            }
            return input;
        }
    };

    template <class ArchTag>
    CATLASS_DEVICE auto get_callbacks(
        Arch::Resource<ArchTag>& resource,
        uint32_t& ub_offset,
        uint32_t compute_length)
    {
        auto ubReduce = resource.ubBuf.template GetBufferByByte<Element>(ub_offset);
        ub_offset += compute_length * sizeof(Element);
        return Callbacks(ubReduce, &params, compute_length);
    }

    Params params;
};

} // namespace Catlass::Epilogue::Fusion

#endif


