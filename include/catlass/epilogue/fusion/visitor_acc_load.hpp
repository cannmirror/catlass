#ifndef CATLASS_EPILOGUE_FUSION_VISITOR_ACC_LOAD_HPP
#define CATLASS_EPILOGUE_FUSION_VISITOR_ACC_LOAD_HPP

#include "catlass/epilogue/fusion/visitor_impl.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"

namespace Catlass::Epilogue::Fusion {

template<class Element>
struct VisitorAccLoad : VisitorImpl<> {
    using VisitorImpl<>::VisitorImpl;

    // 输出元素类型与输出阶段元信息
    using ElementOutput = Element;
    static constexpr VisitStage OUTPUT_STAGE = VisitStage::LOAD;

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
    VisitorAccLoad() {}

    CATLASS_HOST_DEVICE
    VisitorAccLoad(Params const&) {}

    struct Callbacks : EmptyCallbacks {
        AscendC::LocalTensor<Element> ubAcc;
        Params const* params_ptr;
        uint32_t compute_length;
        AscendC::GlobalTensor<Element> gmSubblockC;
        layout::RowMajor layoutSubblockC;

        CATLASS_DEVICE
        Callbacks(AscendC::LocalTensor<Element> ubAcc_,
                 Params const* params_ptr_,
                 uint32_t compute_length_,
                 AscendC::GlobalTensor<Element> const& gmSubblockC_,
                 layout::RowMajor const& layoutSubblockC_)
            : ubAcc(ubAcc_), params_ptr(params_ptr_), compute_length(compute_length_),
              gmSubblockC(gmSubblockC_), layoutSubblockC(layoutSubblockC_) {}

        template <typename... Args>
        CATLASS_DEVICE AscendC::LocalTensor<Element> const& visit(
            MatrixCoord const& globalTileOffset,    // 不使用
            MatrixCoord const& localTileOffset,     // 使用局部坐标
            MatrixCoord const& actualTileShape,
            MatrixCoord const& alignedTileShape,
            uint32_t calCount,
            VisitStage stage,
            Args const&... /*unused*/
        ) {
            if (stage == VisitStage::LOAD) {
                // GM load 使用 actualTileShape，UB layout 使用 alignedTileShape
                auto layoutUb = layout::RowMajor::MakeLayoutInUb<Element>(alignedTileShape);
                using CopyGm2UbT = Epilogue::Tile::CopyGm2Ub<Arch::AtlasA2, Gemm::GemmType<Element, layout::RowMajor>>;
                CopyGm2UbT copyGm2Ub{};

                // 从 gmSubblockC 加载（使用局部坐标，tile 封装处理跨距）
                auto gmTile = gmSubblockC[layoutSubblockC.GetOffset(localTileOffset)];
                auto layoutSrc = layoutSubblockC.GetTileLayout(actualTileShape);
                copyGm2Ub(ubAcc, gmTile, layoutUb, layoutSrc);
            }
            return ubAcc;
        }
    };

    template <class ArchTag>
    CATLASS_DEVICE auto get_callbacks(
        Arch::Resource<ArchTag>& resource,
        uint32_t& ub_offset,
        uint32_t compute_length,
        AscendC::GlobalTensor<Element> const& gmSubblockC,
        layout::RowMajor const& layoutSubblockC
    ) {
        auto ubAcc = resource.ubBuf.template GetBufferByByte<Element>(ub_offset);
        ub_offset += compute_length * sizeof(Element);
        return Callbacks(ubAcc, &params, compute_length, gmSubblockC, layoutSubblockC);
    }

    Params params;
};

} // namespace Catlass::Epilogue::Fusion

#endif
