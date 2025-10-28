#ifndef CATLASS_EPILOGUE_FUSION_VISITOR_COL_REDUCE_HPP
#define CATLASS_EPILOGUE_FUSION_VISITOR_COL_REDUCE_HPP

#include "catlass/epilogue/fusion/visitor_impl.hpp"
#include "catlass/epilogue/fusion/operations.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"

namespace Catlass::Epilogue::Fusion {

// Column reduction over the tile along columns (N dimension):
// reduce each row of the MxN tile into a 1x1 scalar (stored as a length-M UB vector),
// then atomically reduce to a GM column vector. Returns input tensor to allow chaining.
//
// NOTE: First implementation focuses on ReduceFn = Plus (sum) to enable simple atomic accumulation.
//       ReduceFn = Maximum/Minimum can be added similarly when needed.
template <template<class> class ReduceFn, class Element, class Layout = layout::RowMajor, ReduceStrategy Strategy = ReduceStrategy::ATOMIC_REDUCE>
struct VisitorColReduce : VisitorImpl<> {
    using VisitorImpl<>::VisitorImpl;

    // For composition, expose input element type (same as Element)
    using ElementOutput = Element;
    static constexpr VisitStage OUTPUT_STAGE = VisitStage::STORE;

    struct Arguments {
        GM_ADDR ptr_col_out = nullptr; // GM address of M x 1 output column vector
        Layout layout = {};            // layout over (M, 1)
        Element identity = Element(0);
    };

    struct Params {
        GM_ADDR ptr_col_out;
        Layout layout;
        Element identity;

        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GM_ADDR ptr_col_out_, Layout const& layout_, Element const& identity_)
            : ptr_col_out(ptr_col_out_), layout(layout_), identity(identity_) {}
    };

    template <class ProblemShape>
    CATLASS_HOST_DEVICE static constexpr Params
    to_underlying_arguments(ProblemShape const&, Arguments const& args, void*) {
        return Params(args.ptr_col_out, args.layout, args.identity);
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
        return args.ptr_col_out != nullptr;
    }

    CATLASS_HOST_DEVICE
    VisitorColReduce() {}

    CATLASS_HOST_DEVICE
    VisitorColReduce(Params const& params_) : params(params_) {}

    struct Callbacks : EmptyCallbacks {
        AscendC::LocalTensor<Element> ubRowReduce;  // length >= rows
        AscendC::LocalTensor<Element> ubWork;       // work buffer for ReduceSum
        Params const* params_ptr;
        uint32_t compute_length;

        CATLASS_DEVICE
        Callbacks(
            AscendC::LocalTensor<Element> ubRowReduce_,
            AscendC::LocalTensor<Element> ubWork_,
            Params const* params_ptr_,
            uint32_t compute_length_)
            : ubRowReduce(ubRowReduce_), ubWork(ubWork_), params_ptr(params_ptr_), compute_length(compute_length_) {}

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
            static_assert(std::is_same_v<ElementInput, Element>,
                          "VisitorColReduce expects input type == Element. Insert VisitorCast if needed.");

            uint32_t actualRows = actualTileShape.row();
            uint32_t actualCols = actualTileShape.column();
            uint32_t alignedCols = alignedTileShape.column();

            if (stage == VisitStage::COMPUTE) {
                // 使用策略化的计算逻辑
                ColReduceCompute<ReduceFn, Element, Strategy>::compute(
                    ubRowReduce, ubWork, input, actualRows, actualCols, alignedCols);
            }

            if (stage == VisitStage::STORE) {
                // 使用策略化的存储逻辑
                AscendC::GlobalTensor<Element> gmColOut;
                gmColOut.SetGlobalBuffer((__gm__ Element*)(params_ptr->ptr_col_out));
                
                ColReduceCompute<ReduceFn, Element, Strategy>::store(
                    gmColOut, ubRowReduce, globalTileOffset, actualTileShape, params_ptr->layout);
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
        auto ubRowReduce = resource.ubBuf.template GetBufferByByte<Element>(ub_offset);
        ub_offset += compute_length * sizeof(Element);
        auto ubWork = resource.ubBuf.template GetBufferByByte<Element>(ub_offset);
        ub_offset += compute_length * sizeof(Element);
        assert(ub_offset <= ArchTag::UB_SIZE, "ub_offset exceeds ArchTag::UB_SIZE");
        return Callbacks(ubRowReduce, ubWork, &params, compute_length);
    }

    Params params;
};

} // namespace Catlass::Epilogue::Fusion

#endif


