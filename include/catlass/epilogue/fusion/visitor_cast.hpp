#ifndef CATLASS_EPILOGUE_FUSION_VISITOR_CAST_HPP
#define CATLASS_EPILOGUE_FUSION_VISITOR_CAST_HPP

#include "catlass/epilogue/fusion/visitor_impl.hpp"
#include "catlass/epilogue/fusion/operations.hpp"

namespace Catlass::Epilogue::Fusion {

template <class ElementTo, class ElementFrom, AscendC::RoundMode RoundStyle = AscendC::RoundMode::CAST_NONE>
struct VisitorCast : VisitorImpl<> {
    using VisitorImpl<>::VisitorImpl;

    // 输出元素类型与输出阶段元信息
    using ElementOutput = ElementTo;
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
    VisitorCast() {}

    CATLASS_HOST_DEVICE
    VisitorCast(Params const&) {}

    struct Callbacks : EmptyCallbacks {
        AscendC::LocalTensor<ElementTo> ubOut;
        uint32_t compute_length;

        CATLASS_DEVICE
        Callbacks(AscendC::LocalTensor<ElementTo> ubOut_, uint32_t compute_length_)
            : ubOut(ubOut_), compute_length(compute_length_) {}

        template <typename ElementAccumulator, typename ElementInput>
        CATLASS_DEVICE AscendC::LocalTensor<ElementTo> const& visit(
            AscendC::GlobalTensor<ElementAccumulator> const& /*gmSubblockC*/,    // 不使用
            layout::RowMajor const& /*layoutSubblockC*/,           // 不使用
            MatrixCoord const& /*globalTileOffset*/,    // 不使用
            MatrixCoord const& /*localTileOffset*/,     // 不使用
            MatrixCoord const& /*actualTileShape*/,     // 不使用
            MatrixCoord const& /*alignedTileShape*/,   // 不使用
            uint32_t calCount,
            VisitStage stage,
            AscendC::LocalTensor<ElementInput> const& input
        ) {
            static_assert(std::is_same_v<ElementInput, ElementFrom>, "VisitorCast: input type mismatch");
            static_assert(!std::is_same_v<ElementInput, ElementTo>, "VisitorCast: no need to cast");
            if (stage == VisitStage::COMPUTE) {
                Cast<ElementTo, ElementInput, RoundStyle>{}(ubOut, compute_length, input);
            }
            return ubOut;
        }
    };

    template <class ArchTag>
    CATLASS_DEVICE auto get_callbacks(
        Arch::Resource<ArchTag>& resource,
        uint32_t& ub_offset,
        uint32_t compute_length
    ) {
        auto ubOut = resource.ubBuf.template GetBufferByByte<ElementTo>(ub_offset);
        ub_offset += compute_length * sizeof(ElementTo);
        assert(ub_offset <= ArchTag::UB_SIZE, "ub_offset exceeds ArchTag::UB_SIZE");
        return Callbacks(ubOut, compute_length);
    }

    Params params;
};

} // namespace Catlass::Epilogue::Fusion

#endif


