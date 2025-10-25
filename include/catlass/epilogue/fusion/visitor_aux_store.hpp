#ifndef CATLASS_EPILOGUE_FUSION_VISITOR_AUX_STORE_HPP
#define CATLASS_EPILOGUE_FUSION_VISITOR_AUX_STORE_HPP

#include "catlass/epilogue/fusion/visitor_impl.hpp"
#include "catlass/epilogue/fusion/operations.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"

namespace Catlass::Epilogue::Fusion {

template<
  class Element,
  class Layout = layout::RowMajor
>
struct VisitorAuxStore : VisitorImpl<> {
    using VisitorImpl<>::VisitorImpl;

    // 输出元素类型与输出阶段元信息
    using ElementOutput = Element;
    static constexpr VisitStage OUTPUT_STAGE = VisitStage::STORE;

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
    VisitorAuxStore() {}

    CATLASS_HOST_DEVICE
    VisitorAuxStore(Params const& params_) : params(params_) {}

    struct Callbacks : EmptyCallbacks {
        Params const* params_ptr;

        CATLASS_DEVICE
        Callbacks(Params const* params_ptr_)
            : params_ptr(params_ptr_) {}

        template <typename ElementAccumulator, typename ElementInput>
        CATLASS_DEVICE AscendC::LocalTensor<ElementInput> const& visit(
            AscendC::GlobalTensor<ElementAccumulator> const& /*gmSubblockC*/,
            layout::RowMajor const& /*layoutSubblockC*/,
            MatrixCoord const& globalTileOffset,
            MatrixCoord const& localTileOffset,  // 新增参数（不使用）
            MatrixCoord const& actualTileShape,
            MatrixCoord const& alignedTileShape,
            uint32_t calCount,
            VisitStage stage,
            AscendC::LocalTensor<ElementInput> const& input
        ) {
            static_assert(std::is_same_v<ElementInput, Element>,
                          "VisitorAuxStore: element type mismatch. Insert VisitorCast<...> before store.");
            if (stage == VisitStage::STORE) {
                // 写回 GM（使用全局坐标，tile 封装处理跨距）
                if (params_ptr->ptr_aux != nullptr) {
                    // GM store 使用 actualTileShape，UB layout 使用 alignedTileShape
                    auto layoutUb = layout::RowMajor::MakeLayoutInUb<Element>(alignedTileShape);
                    using CopyUb2GmT = Epilogue::Tile::CopyUb2Gm<Arch::AtlasA2, Gemm::GemmType<Element, layout::RowMajor>>;
                    CopyUb2GmT copyUb2Gm{};
                    AscendC::GlobalTensor<Element> gmAux;
                    gmAux.SetGlobalBuffer((__gm__ Element*)(params_ptr->ptr_aux));
                    auto gmTile = gmAux[params_ptr->layout.GetOffset(globalTileOffset)];
                    auto layoutDst = params_ptr->layout.GetTileLayout(actualTileShape);
                    copyUb2Gm(gmTile, input, layoutDst, layoutUb);
                }

            }
            // 透传返回输入以便继续参与 EVT 组合
            return input;
        }
    };

    template <class ArchTag>
    CATLASS_DEVICE auto get_callbacks(
        Arch::Resource<ArchTag>& resource,
        uint32_t& ub_offset,
        uint32_t compute_length
    ) {
        return Callbacks(&params);
    }

    Params params;
};

} // namespace Catlass::Epilogue::Fusion

#endif



