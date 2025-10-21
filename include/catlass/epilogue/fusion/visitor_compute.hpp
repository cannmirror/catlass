#ifndef CATLASS_EPILOGUE_FUSION_VISITOR_COMPUTE_HPP
#define CATLASS_EPILOGUE_FUSION_VISITOR_COMPUTE_HPP

#include "catlass/epilogue/fusion/visitor_impl.hpp"
#include "catlass/epilogue/fusion/operations.hpp"

namespace Catlass::Epilogue::Fusion {

template<
  template <class> class ComputeFn,
  class ElementCompute
>
struct VisitorCompute : VisitorImpl<> {
    using VisitorImpl<>::VisitorImpl;

    // 输出元素类型与输出阶段元信息
    using ElementOutput = ElementCompute;
    static constexpr VisitStage OUTPUT_STAGE = VisitStage::COMPUTE;

    struct Arguments {};
    struct Params {};

    template <class ProblemShape>
    CATLASS_HOST_DEVICE static constexpr Params
    to_underlying_arguments(ProblemShape const&, Arguments const&, void*) {
        return Params();
    }

    template <class ProblemShape>
    CATLASS_HOST_DEVICE static size_t
    get_workspace_size(ProblemShape const&, Arguments const&) {
        return 0;
    }

    template <class ProblemShape>
    CATLASS_HOST_DEVICE static bool
    can_implement(ProblemShape const&, Arguments const&) {
        return true;
    }

    CATLASS_HOST_DEVICE
    VisitorCompute() {}

    CATLASS_HOST_DEVICE
    VisitorCompute(Params const&) {}

    struct Callbacks : EmptyCallbacks {
        AscendC::LocalTensor<ElementCompute> ubOut;
        uint32_t compute_length;

        CATLASS_DEVICE
        Callbacks(AscendC::LocalTensor<ElementCompute> ubOut_,
                 uint32_t compute_length_)
            : ubOut(ubOut_), compute_length(compute_length_) {}

        template <typename... ElementInputs>
        CATLASS_DEVICE AscendC::LocalTensor<ElementCompute> const& visit(
            MatrixCoord const& globalTileOffset,    // 不使用
            MatrixCoord const& localTileOffset,     // 新增参数（不使用）
            MatrixCoord const& actualTileShape,     // 不使用
            MatrixCoord const& alignedTileShape,    // 不使用
            uint32_t calCount,
            VisitStage stage,
            AscendC::LocalTensor<ElementInputs> const&... inputs
        ) {
            if (stage == VisitStage::COMPUTE) {
                constexpr bool all_inputs_match = (std::is_same_v<ElementInputs, ElementCompute> && ...);
                static_assert(all_inputs_match,
                              "VisitorCompute: input element types must equal ElementCompute. Insert VisitorCast if needed.");

                ComputeFn<ElementCompute> compute_fn{};
                compute_fn(ubOut, inputs..., compute_length);
            }
            return ubOut;
        }
    };

    template <class ArchTag>
    CATLASS_DEVICE auto get_callbacks(
        Arch::Resource<ArchTag>& resource,
        uint32_t& ub_offset,
        uint32_t compute_length,
        AscendC::GlobalTensor<half> const&,
        layout::RowMajor const&
    ) {
        auto ubOut = resource.ubBuf.template GetBufferByByte<ElementCompute>(ub_offset);
        ub_offset += compute_length * sizeof(ElementCompute);
        return Callbacks(ubOut, compute_length);
    }
};

} // namespace Catlass::Epilogue::Fusion

#endif
