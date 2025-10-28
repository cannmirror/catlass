#ifndef CATLASS_EPILOGUE_FUSION_VISITOR_SCALAR_REDUCE_HPP
#define CATLASS_EPILOGUE_FUSION_VISITOR_SCALAR_REDUCE_HPP

#include <type_traits>

#include "catlass/epilogue/fusion/operations.hpp"
#include "catlass/epilogue/fusion/visitor_impl.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"

namespace Catlass::Epilogue::Fusion {

// Reduce the entire MxN tile to a single scalar and accumulate to GM.
// COMPUTE stage: perform scalar reduction and store in UB.
// STORE stage: atomically write the scalar result back to GM.
template <template<class> class ReduceFn, class Element, class Layout = layout::RowMajor, ReduceStrategy Strategy = ReduceStrategy::ATOMIC_REDUCE>
struct VisitorScalarReduce : VisitorImpl<> {
    using VisitorImpl<>::VisitorImpl;

    using ElementOutput = Element;
    static constexpr VisitStage OUTPUT_STAGE = VisitStage::STORE;

    struct Arguments {
        GM_ADDR ptr_scalar_out = nullptr; // GM address of scalar output
        Layout layout = {};               // layout describing scalar location
        Element identity = Element(0);
    };

    struct Params {
        GM_ADDR ptr_scalar_out;
        Layout layout;
        Element identity;

        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GM_ADDR ptr_scalar_out_, Layout const& layout_, Element const& identity_)
            : ptr_scalar_out(ptr_scalar_out_), layout(layout_), identity(identity_) {}
    };

    template <class ProblemShape>
    CATLASS_HOST_DEVICE static constexpr Params
    to_underlying_arguments(ProblemShape const&, Arguments const& args, void*) {
        return Params(args.ptr_scalar_out, args.layout, args.identity);
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
        return args.ptr_scalar_out != nullptr;
    }

    CATLASS_HOST_DEVICE
    VisitorScalarReduce() {}

    CATLASS_HOST_DEVICE
    explicit VisitorScalarReduce(Params const& params_) : params(params_) {}

    struct Callbacks : EmptyCallbacks {
        AscendC::LocalTensor<Element> ubScalar; // buffer to hold the reduced scalar
        AscendC::LocalTensor<Element> ubWork; // work buffer
        Params const* params_ptr;
        uint32_t compute_length;

        CATLASS_DEVICE
        Callbacks(
            AscendC::LocalTensor<Element> ubScalar_,
            AscendC::LocalTensor<Element> ubWork_,
            Params const* params_ptr_,
            uint32_t compute_length_)
            : ubScalar(ubScalar_), ubWork(ubWork_), params_ptr(params_ptr_), compute_length(compute_length_) {}

        template <typename ElementAccumulator, typename ElementInput>
        CATLASS_DEVICE AscendC::LocalTensor<ElementInput> const& visit(
            AscendC::GlobalTensor<ElementAccumulator> const& /*gmSubblockC*/,
            layout::RowMajor const& /*layoutSubblockC*/,
            MatrixCoord const& /*globalTileOffset*/,
            MatrixCoord const& /*localTileOffset*/,
            MatrixCoord const& actualTileShape,
            MatrixCoord const& alignedTileShape,
            uint32_t /*calCount*/,
            VisitStage stage,
            AscendC::LocalTensor<ElementInput> const& input)
        {
            static_assert(std::is_same_v<ElementInput, Element>,
                          "VisitorScalarReduce expects input type == Element. Insert VisitorCast if needed.");

            uint32_t actualRows = actualTileShape.row();
            uint32_t actualCols = actualTileShape.column();
            uint32_t alignedCols = alignedTileShape.column();

            if (stage == VisitStage::COMPUTE) {
                // 使用策略化的计算逻辑
                ScalarReduceCompute<ReduceFn, Element, Strategy>::compute(
                    ubScalar, ubWork, input, params_ptr->identity, actualRows, actualCols);
            }

            if (stage == VisitStage::STORE) {
                // 使用策略化的存储逻辑
                AscendC::GlobalTensor<Element> gmOut;
                gmOut.SetGlobalBuffer((__gm__ Element*)(params_ptr->ptr_scalar_out));
                
                ScalarReduceCompute<ReduceFn, Element, Strategy>::store(
                    gmOut, ubScalar, params_ptr->layout);
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
        auto ubScalar = resource.ubBuf.template GetBufferByByte<Element>(ub_offset);
        ub_offset += compute_length * sizeof(Element);
        auto ubWork = resource.ubBuf.template GetBufferByByte<Element>(ub_offset);
        ub_offset += compute_length * sizeof(Element);
        return Callbacks(ubScalar, ubWork, &params, compute_length);
    }

    Params params;
};

} // namespace Catlass::Epilogue::Fusion

#endif
