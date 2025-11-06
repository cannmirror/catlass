#include "mla_kernel.h"

#include "catlass/catlass.hpp"

#include "mla.h"
#include "mla_tp1_spec.h"
template <class DataType>
CATLASS_GLOBAL void
MLA(uint64_t fftsAddr,
    GM_ADDR q,
    GM_ADDR qRope,
    GM_ADDR k,
    GM_ADDR kRope,
    GM_ADDR blockTables,
    GM_ADDR o,
    GM_ADDR s,
    GM_ADDR p,
    GM_ADDR oTmp,
    GM_ADDR oUpdate,
    GM_ADDR oCoreTmp,
    GM_ADDR l,
    GM_ADDR tiling)
{
    // Set FFTS address
    AscendC::SetSyncBaseAddr(fftsAddr);

    using ArchTag = Arch::AtlasA2;
    using ElementQ = DataType;
    using LayoutQ = layout::RowMajor;
    using ElementK = DataType;
    using LayoutK = layout::ColumnMajor;
    using ElementV = DataType;
    using LayoutV = layout::RowMajor;
    using ElementS = float;
    using LayoutS = layout::RowMajor;
    using ElementP = DataType;
    using LayoutP = layout::RowMajor;
    using ElementO = DataType;
    using LayoutO = layout::RowMajor;
    using ElementMask = DataType;
    using LayoutMask = layout::RowMajor;
    using ElementOTmp = float;
    using LayoutOTmp = layout::RowMajor;
    using ElementUpdate = float;
    using LayoutUpdate = layout::RowMajor;

    // L1TileShape::K must be embdding
    using L1TileShape = GemmShape<128, 128, 576>;
    using L0TileShape = L1TileShape;

    // GEMM Block模块，实现Flash MLA的Q * K^T
    using DispatchPolicyQK = Gemm::MmadAtlasA2MLAQK;
    using QType = Gemm::GemmType<ElementQ, LayoutQ>;
    using KType = Gemm::GemmType<ElementK, LayoutK>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using BlockMmadQK = Gemm::Block::BlockMmad<DispatchPolicyQK, L1TileShape, L0TileShape, QType, KType, SType>;

    // Epilogue Block模块，实现Flash MLA中当前S基块的softmax
    using PType = Gemm::GemmType<ElementP, LayoutP>;
    using MaskType = Gemm::GemmType<ElementMask, LayoutMask>;
    using EpilogueMLASoftmax =
        Epilogue::Block::BlockEpilogue<Epilogue::EpilogueAtlasA2MLASoftmax, PType, SType, MaskType>;

    // GEMM Block模块，实现Flash MLA的P * V
    using DispatchPolicyPV = Gemm::MmadAtlasA2MLAPV;
    using VType = Gemm::GemmType<ElementV, LayoutV>;
    using OTmpType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
    using BlockMmadPV = Gemm::Block::BlockMmad<DispatchPolicyPV, L1TileShape, L0TileShape, PType, VType, OTmpType>;

    // Epilogue Block模块，实现Flash MLA中当前O基块的更新
    using OType = Gemm::GemmType<ElementO, LayoutO>;
    using OUpdateType = Gemm::GemmType<ElementUpdate, LayoutUpdate>;
    using EpilogueMLARescaleO =
        Epilogue::Block::BlockEpilogue<Epilogue::EpilogueAtlasA2MLARescaleO, OType, OUpdateType, OTmpType>;

    // Epilogue Block模块，实现Flash MLA中flash decoding
    using OType = Gemm::GemmType<ElementO, LayoutO>;
    using lType = Gemm::GemmType<ElementUpdate, LayoutUpdate>;
    constexpr uint32_t ComputeEleNum = 6144;
    using EpilogueMLAFDRescaleO =
        Epilogue::Block::BlockEpilogue<Epilogue::EpilogueAtlasA2MLAFDRescaleO<ComputeEleNum>, OType, lType>;

    // Kernel level
    using MLAKernel =
        MLAKernel<BlockMmadQK, BlockMmadPV, EpilogueMLASoftmax, EpilogueMLARescaleO, EpilogueMLAFDRescaleO>;
    typename MLAKernel::Params params{q, qRope, k, kRope, blockTables, o, s, p, oTmp, oUpdate, oCoreTmp, l, tiling};

    // call kernel
    MLAKernel mla;
    mla(params);
}
template <class DataType>
CATLASS_GLOBAL void MLATp1Spec(
    uint64_t fftsAddr,
    GM_ADDR q,
    GM_ADDR qRope,
    GM_ADDR k,
    GM_ADDR kRope,
    GM_ADDR blockTables,
    GM_ADDR o,
    GM_ADDR s,
    GM_ADDR p,
    GM_ADDR oTmp,
    GM_ADDR oUpdate,
    GM_ADDR oCoreTmp,
    GM_ADDR l,
    GM_ADDR tiling
)
{
    // Set FFTS address
    AscendC::SetSyncBaseAddr(fftsAddr);

    using ArchTag = Arch::AtlasA2;
    using ElementQ = DataType;
    using LayoutQ = layout::RowMajor;
    using ElementK = DataType;
    using LayoutK = layout::ColumnMajor;
    using ElementV = DataType;
    using LayoutV = layout::RowMajor;
    using ElementS = float;
    using LayoutS = layout::RowMajor;
    using ElementP = DataType;
    using LayoutP = layout::RowMajor;
    using ElementO = DataType;
    using LayoutO = layout::RowMajor;
    using ElementMask = DataType;
    using LayoutMask = layout::RowMajor;
    using ElementOTmp = float;
    using LayoutOTmp = layout::RowMajor;
    using ElementUpdate = float;
    using LayoutUpdate = layout::RowMajor;

    // L1TileShape::K must be embdding
    using L1TileShape = GemmShape<128, 128, 576>;
    using L0TileShape = L1TileShape;

    // GEMM Block模块，实现Flash MLA的Q * K^T
    using DispatchPolicyQK = Gemm::MmadAtlasA2MLAQKTp1Spec;
    using QType = Gemm::GemmType<ElementQ, LayoutQ>;
    using KType = Gemm::GemmType<ElementK, LayoutK>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using BlockMmadQK = Gemm::Block::BlockMmad<DispatchPolicyQK, L1TileShape, L0TileShape, QType, KType, SType>;

    // Epilogue Block模块，实现Flash MLA中当前S基块的softmax
    using PType = Gemm::GemmType<ElementP, LayoutP>;
    using MaskType = Gemm::GemmType<ElementMask, LayoutMask>;
    using EpilogueMLASoftmax =
        Epilogue::Block::BlockEpilogue<Epilogue::EpilogueAtlasA2MLATP1Softmax, PType, SType, MaskType>;

    // GEMM Block模块，实现Flash MLA的P * V
    using DispatchPolicyPV = Gemm::MmadAtlasA2MLAPVTp1Spec;
    using VType = Gemm::GemmType<ElementV, LayoutV>;
    using OTmpType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
    using BlockMmadPV = Gemm::Block::BlockMmad<DispatchPolicyPV, L1TileShape, L0TileShape, PType, VType, OTmpType>;

    // Epilogue Block模块，实现Flash MLA中当前O基块的更新
    using OType = Gemm::GemmType<ElementO, LayoutO>;
    using OUpdateType = Gemm::GemmType<ElementUpdate, LayoutUpdate>;
    using EpilogueMLARescaleO =
        Epilogue::Block::BlockEpilogue<Epilogue::EpilogueAtlasA2MLATP1RescaleO, OType, OUpdateType, OTmpType>;

    // Epilogue Block模块，实现Flash MLA中flash decoding
    using OType = Gemm::GemmType<ElementO, LayoutO>;
    using lType = Gemm::GemmType<ElementUpdate, LayoutUpdate>;
    constexpr uint32_t ComputeEleNum = 6144;
    using EpilogueMLAFDRescaleO =
        Epilogue::Block::BlockEpilogue<Epilogue::EpilogueAtlasA2MLAFDRescaleO<ComputeEleNum>, OType, lType>;

    // Kernel level
    using MLAKernel =
        MLAKernelTp1Spec<BlockMmadQK, BlockMmadPV, EpilogueMLASoftmax, EpilogueMLARescaleO, EpilogueMLAFDRescaleO>;
    typename MLAKernel::Params params{q, qRope, k, kRope, blockTables, o, s, p, oTmp, oUpdate, oCoreTmp, l, tiling};

    // call kernel
    MLAKernel mla;
    mla(params);
}

void MLADevice(
    uint8_t blockNum,
    aclrtStream stream,
    aclDataType dataType,
    bool isTp1Spec,
    uint64_t fftsAddr,
    uint8_t *q,
    uint8_t *qRope,
    uint8_t *k,
    uint8_t *kRope,
    uint8_t *blockTables,
    uint8_t *o,
    uint8_t *s,
    uint8_t *p,
    uint8_t *oTmp,
    uint8_t *oUpdate,
    uint8_t *oCoreTmp,
    uint8_t *l,
    uint8_t *tiling
)
{
    if (isTp1Spec) {
        if (dataType == ACL_FLOAT16) {
            MLATp1Spec<half><<<blockNum, nullptr, stream>>>(
                fftsAddr, q, qRope, k, kRope, blockTables, o, s, p, oTmp, oUpdate, oCoreTmp, l, tiling
            );
        } else if (dataType == ACL_BF16) {
            MLATp1Spec<bfloat16_t><<<blockNum, nullptr, stream>>>(
                fftsAddr, q, qRope, k, kRope, blockTables, o, s, p, oTmp, oUpdate, oCoreTmp, l, tiling
            );
        }
    } else {
        if (dataType == ACL_FLOAT16) {
            MLA<half><<<blockNum, nullptr, stream>>>(
                fftsAddr, q, qRope, k, kRope, blockTables, o, s, p, oTmp, oUpdate, oCoreTmp, l, tiling
            );
        } else if (dataType == ACL_BF16) {
            MLA<bfloat16_t><<<blockNum, nullptr, stream>>>(
                fftsAddr, q, qRope, k, kRope, blockTables, o, s, p, oTmp, oUpdate, oCoreTmp, l, tiling
            );
        }
    }
}
