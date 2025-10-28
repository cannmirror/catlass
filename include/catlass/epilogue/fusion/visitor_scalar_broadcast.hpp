#ifndef CATLASS_EPILOGUE_FUSION_VISITOR_SCALAR_BROADCAST_HPP
#define CATLASS_EPILOGUE_FUSION_VISITOR_SCALAR_BROADCAST_HPP

#include "catlass/epilogue/fusion/visitor_impl.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"

namespace Catlass::Epilogue::Fusion {

// Broadcast a single scalar from GM to every element of the current tile.
// LOAD stage: load one scalar value into UB.
// COMPUTE stage: duplicate the scalar across the tile buffer and return it.
template <class Element, class Layout = layout::RowMajor>
struct VisitorScalarBroadcast : VisitorImpl<> {
    using VisitorImpl<>::VisitorImpl;

    using ElementOutput = Element;
    static constexpr VisitStage OUTPUT_STAGE = VisitStage::COMPUTE;

    struct Arguments {
        GM_ADDR ptr_scalar = nullptr; // GM address of scalar value
        Layout layout = {};           // layout describing the scalar position (default 1x1)
    };

    struct Params {
        GM_ADDR ptr_scalar;
        Layout layout;

        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GM_ADDR ptr_scalar_, Layout const& layout_)
            : ptr_scalar(ptr_scalar_), layout(layout_) {}
    };

    template <class ProblemShape>
    CATLASS_HOST_DEVICE static constexpr Params
    to_underlying_arguments(ProblemShape const&, Arguments const& args, void*) {
        return Params(args.ptr_scalar, args.layout);
    }

    template <class ProblemShape>
    CATLASS_HOST_DEVICE static size_t
    get_workspace_size(ProblemShape const&, Arguments const&) {
        return 0;
    }

    template <class ProblemShape>
    CATLASS_HOST_DEVICE static bool
    can_implement(ProblemShape const&, Arguments const& args) {
        return args.ptr_scalar != nullptr;
    }

    CATLASS_HOST_DEVICE
    VisitorScalarBroadcast() {}

    CATLASS_HOST_DEVICE
    explicit VisitorScalarBroadcast(Params const& params_) : params(params_) {}

    struct Callbacks : EmptyCallbacks {
        AscendC::LocalTensor<Element> ubOut; // size >= compute_length
        Params const* params_ptr;
        uint32_t compute_length;

        CATLASS_DEVICE
        Callbacks(
            AscendC::LocalTensor<Element> ubOut_,
            Params const* params_ptr_,
            uint32_t compute_length_)
            : ubOut(ubOut_), params_ptr(params_ptr_), compute_length(compute_length_) {}

        template <typename ElementAccumulator, typename... Args>
        CATLASS_DEVICE AscendC::LocalTensor<Element> const& visit(
            AscendC::GlobalTensor<ElementAccumulator> const& /*gmSubblockC*/,
            layout::RowMajor const& /*layoutSubblockC*/,
            MatrixCoord const& /*globalTileOffset*/,
            MatrixCoord const& /*localTileOffset*/,
            MatrixCoord const& actualTileShape,
            MatrixCoord const& alignedTileShape,
            uint32_t /*calCount*/,
            VisitStage stage,
            Args const&... /*unused*/)
        {
            uint32_t actualRows = actualTileShape.row();
            uint32_t actualCols = actualTileShape.column();
            uint32_t alignedCols = alignedTileShape.column();

            if (stage == VisitStage::LOAD) {
                auto layoutUbScalar = layout::RowMajor::MakeLayoutInUb<Element>(MatrixCoord{1u, 1u});
                using CopyGm2UbT = Epilogue::Tile::CopyGm2Ub<Arch::AtlasA2, Gemm::GemmType<Element, layout::RowMajor>>;
                CopyGm2UbT copyGm2Ub{};

                AscendC::GlobalTensor<Element> gmScalar;
                gmScalar.SetGlobalBuffer((__gm__ Element*)(params_ptr->ptr_scalar));
                auto gmTile = gmScalar[params_ptr->layout.GetOffset(MatrixCoord{0u, 0u})];
                auto layoutSrc = params_ptr->layout.GetTileLayout(MatrixCoord{1u, 1u});
                copyGm2Ub(ubOut, gmTile, layoutUbScalar, layoutSrc);
            }

            if (stage == VisitStage::COMPUTE) {
                AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
                Element v = ubOut.GetValue(0);
                for (uint32_t r = 0; r < actualRows; ++r) {
                    AscendC::Duplicate<Element>(ubOut[r * alignedCols], v, actualCols);
                }
            }
            return ubOut;
        }
    };

    template <class ArchTag>
    CATLASS_DEVICE auto get_callbacks(
        Arch::Resource<ArchTag>& resource,
        uint32_t& ub_offset,
        uint32_t compute_length)
    {
        auto ubOut = resource.ubBuf.template GetBufferByByte<Element>(ub_offset);
        ub_offset += compute_length * sizeof(Element);
        return Callbacks(ubOut, &params, compute_length);
    }

    Params params;
};

} // namespace Catlass::Epilogue::Fusion

#endif

