#ifndef CATLASS_EPILOGUE_FUSION_VISITOR_AUX_LOAD_HPP
#define CATLASS_EPILOGUE_FUSION_VISITOR_AUX_LOAD_HPP

#include "catlass/epilogue/fusion/visitor_impl.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"

namespace Catlass::Epilogue::Fusion {

template<class Element, class Layout = layout::RowMajor>
struct VisitorAuxLoad : VisitorImpl<> {
    using VisitorImpl<>::VisitorImpl;

    // 输出元素类型与输出阶段元信息
    using ElementOutput = Element;
    static constexpr VisitStage OUTPUT_STAGE = VisitStage::LOAD;

    struct Arguments {
        GM_ADDR ptr_aux = nullptr;
        Layout layout = {};
    };

    struct Params {
        GM_ADDR ptr_aux;
        Layout layout;

        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GM_ADDR ptr_aux_, Layout const& layout_)
            : ptr_aux(ptr_aux_), layout(layout_) {}
    };

    template <class ProblemShape>
    CATLASS_HOST_DEVICE static constexpr Params
    to_underlying_arguments(ProblemShape const&, Arguments const& args, void*) {
        return Params(args.ptr_aux, args.layout);
    }

    template <class ProblemShape>
    CATLASS_HOST_DEVICE static size_t
    get_workspace_size(ProblemShape const&, Arguments const&) {
        return 0;
    }

    template <class ProblemShape>
    CATLASS_HOST_DEVICE static bool
    can_implement(ProblemShape const&, Arguments const& args) {
        return args.ptr_aux != nullptr;
    }

    CATLASS_HOST_DEVICE
    VisitorAuxLoad() {}

    CATLASS_HOST_DEVICE
    VisitorAuxLoad(Params const& params_) : params(params_) {}

    struct Callbacks : EmptyCallbacks {
        AscendC::LocalTensor<Element> ubAux;
        Params const* params_ptr;
        uint32_t compute_length;

        CATLASS_DEVICE
        Callbacks(AscendC::LocalTensor<Element> ubAux_,
                 Params const* params_ptr_,
                 uint32_t compute_length_)
            : ubAux(ubAux_), params_ptr(params_ptr_), compute_length(compute_length_) {}

        template <typename ElementAccumulator, typename... Args>
        CATLASS_DEVICE AscendC::LocalTensor<Element> const& visit(
            AscendC::GlobalTensor<ElementAccumulator> const& /*gmSubblockC*/,
            layout::RowMajor const& /*layoutSubblockC*/,
            MatrixCoord const& globalTileOffset,
            MatrixCoord const& localTileOffset,  // 新增参数（不使用）
            MatrixCoord const& actualTileShape,
            MatrixCoord const& alignedTileShape,
            uint32_t calCount,
            VisitStage stage,
            Args const&... /*unused*/
        ) {
            if (stage == VisitStage::LOAD) {
                // GM load 使用 actualTileShape，UB layout 使用 alignedTileShape
                auto layoutUb = layout::RowMajor{alignedTileShape.row(), actualTileShape.column(), alignedTileShape.column()};
                using CopyGm2UbT = Epilogue::Tile::CopyGm2Ub<Arch::AtlasA2, Gemm::GemmType<Element, layout::RowMajor>>;
                CopyGm2UbT copyGm2Ub{};

                // 从用户提供的 ptr_aux 加载（使用全局坐标，tile 封装处理跨距）
                AscendC::GlobalTensor<Element> gmAux;
                gmAux.SetGlobalBuffer((__gm__ Element*)(params_ptr->ptr_aux));
                auto gmTile = gmAux[params_ptr->layout.GetOffset(globalTileOffset)];
                auto layoutSrc = params_ptr->layout.GetTileLayout(actualTileShape);
                copyGm2Ub(ubAux, gmTile, layoutUb, layoutSrc);
            }
            return ubAux;
        }
    };

    template <class ArchTag>
    CATLASS_DEVICE auto get_callbacks(
        Arch::Resource<ArchTag>& resource,
        uint32_t& ub_offset,
        uint32_t compute_length
    ) {
        auto ubAux = resource.ubBuf.template GetBufferByByte<Element>(ub_offset);
        ub_offset += compute_length * sizeof(Element);
        assert(ub_offset <= ArchTag::UB_SIZE, "ub_offset exceeds ArchTag::UB_SIZE");
        return Callbacks(ubAux, &params, compute_length);
    }

    Params params;
};


} // namespace Catlass::Epilogue::Fusion

#endif



