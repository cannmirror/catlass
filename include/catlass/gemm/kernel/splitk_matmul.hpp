/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_SPLITK_MATMUL_HPP
#define CATLASS_GEMM_KERNEL_SPLITK_MATMUL_HPP

#include <cmath>
#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

namespace Catlass::Gemm::Kernel {

template<
    class ArchTag_,
    class ElementAccumulator_,
    class ElementOut_,
    uint32_t COMPUTE_LENGTH
>
struct ReduceAdd {
    using ArchTag = ArchTag_;
    using ElementAccumulator = ElementAccumulator_;
    using ElementOut = ElementOut_;

    CATLASS_DEVICE
    ReduceAdd(Arch::Resource<ArchTag> &resource)
    {
        int64_t bufferOffset = 0;
        for (uint32_t i = 0; i < BUFFER_NUM; i++) {
            inputBuffer[i] = resource.ubBuf.template GetBufferByByte<ElementAccumulator>(bufferOffset);
            bufferOffset += COMPUTE_LENGTH * sizeof(ElementAccumulator);
            accumulatorBuffer[i] = resource.ubBuf.template GetBufferByByte<ElementAccumulator>(bufferOffset);
            bufferOffset += COMPUTE_LENGTH * sizeof(ElementAccumulator);
            outputBuffer[i] = resource.ubBuf.template GetBufferByByte<ElementOut>(bufferOffset);
            bufferOffset += COMPUTE_LENGTH * sizeof(ElementOut);
        }
    }

    CATLASS_DEVICE
    void Gm2Ub(AscendC::LocalTensor<ElementAccumulator> const &dst,
        AscendC::GlobalTensor<ElementAccumulator> const &src,
        uint32_t dataNum)
    {
        AscendC::DataCopyExtParams dataCopyParams(1, dataNum * sizeof(ElementAccumulator), 0, 0, 0);
        AscendC::DataCopyPadExtParams<ElementAccumulator> padParams(false, 0, 0, 0);
        AscendC::DataCopyPad(dst, src, dataCopyParams, padParams);
    }

    CATLASS_DEVICE
    void Ub2Gm(AscendC::GlobalTensor<ElementOut> const &dst,
        AscendC::LocalTensor<ElementOut> const &src,
        uint32_t dataNum)
    {
        AscendC::DataCopyExtParams dataCopyParams(1, dataNum * sizeof(ElementOut), 0, 0, 0);
        AscendC::DataCopyPad(dst, src, dataCopyParams);
    }

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementOut> const &dst,
        AscendC::GlobalTensor<ElementAccumulator> const &src,
        uint64_t elementCount, uint32_t splitkFactor)
    {
        // The vec mte processes 256 bytes of data at a time.
        constexpr uint32_t ELE_PER_VECOTR_BLOCK = 256 / sizeof(ElementAccumulator);
        uint32_t aivNum = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
        uint32_t aivId = AscendC::GetBlockIdx();
        uint64_t taskPerAiv =
            (elementCount / aivNum + ELE_PER_VECOTR_BLOCK - 1) / ELE_PER_VECOTR_BLOCK * ELE_PER_VECOTR_BLOCK;
        if (taskPerAiv == 0) taskPerAiv = ELE_PER_VECOTR_BLOCK;
        uint32_t tileLen;
        if (taskPerAiv > COMPUTE_LENGTH) {
            tileLen = COMPUTE_LENGTH;
        } else {
            tileLen = taskPerAiv;
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputEventIds[0]);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputEventIds[1]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputEventIds[0]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputEventIds[1]);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(accumulatorEventIds[0]);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(accumulatorEventIds[1]);

        uint32_t loops = (elementCount + tileLen - 1) / tileLen;
        for (uint32_t loopIdx = aivId; loopIdx < loops; loopIdx += aivNum) {
            uint32_t actualTileLen = tileLen;
            if (loopIdx == loops - 1) {
                actualTileLen = elementCount - loopIdx * tileLen;
            }

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(accumulatorEventIds[bufferIndex]);
            Gm2Ub(accumulatorBuffer[bufferIndex], src[loopIdx * tileLen], actualTileLen);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(accumulatorEventIds[bufferIndex]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(accumulatorEventIds[bufferIndex]);

            for (uint32_t sliceIdx = 1; sliceIdx < splitkFactor; ++sliceIdx) {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputEventIds[bufferIndex]);
                Gm2Ub(inputBuffer[bufferIndex],
                    src[sliceIdx * elementCount + loopIdx * tileLen], actualTileLen);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputEventIds[bufferIndex]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputEventIds[bufferIndex]);

                AscendC::Add(accumulatorBuffer[bufferIndex],
                    accumulatorBuffer[bufferIndex], inputBuffer[bufferIndex], actualTileLen);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputEventIds[bufferIndex]);
            }
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputEventIds[bufferIndex]);
            if constexpr (!std::is_same_v<ElementAccumulator, ElementOut>) {
                if constexpr (std::is_same_v<ElementOut, half>) {
                    AscendC::Cast(outputBuffer[bufferIndex],
                        accumulatorBuffer[bufferIndex], AscendC::RoundMode::CAST_NONE, actualTileLen);
                } else {
                    AscendC::Cast(outputBuffer[bufferIndex],
                        accumulatorBuffer[bufferIndex], AscendC::RoundMode::CAST_RINT, actualTileLen);
                }
            } else {
                AscendC::DataCopy(outputBuffer[bufferIndex], accumulatorBuffer[bufferIndex], tileLen);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(accumulatorEventIds[bufferIndex]);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputEventIds[bufferIndex]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputEventIds[bufferIndex]);
            Ub2Gm(dst[loopIdx * tileLen], outputBuffer[bufferIndex], actualTileLen);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputEventIds[bufferIndex]);

            bufferIndex = (bufferIndex + 1) % BUFFER_NUM;
        }

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputEventIds[0]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputEventIds[1]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputEventIds[0]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputEventIds[1]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(accumulatorEventIds[0]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(accumulatorEventIds[1]);
    }

private:
    static const uint32_t BUFFER_NUM = 2;
    AscendC::LocalTensor<ElementAccumulator> inputBuffer[BUFFER_NUM];
    AscendC::LocalTensor<ElementAccumulator> accumulatorBuffer[BUFFER_NUM];
    AscendC::LocalTensor<ElementOut> outputBuffer[BUFFER_NUM];
    AscendC::TEventID inputEventIds[BUFFER_NUM] = {EVENT_ID0, EVENT_ID1};
    AscendC::TEventID accumulatorEventIds[BUFFER_NUM] = {EVENT_ID2, EVENT_ID3};
    AscendC::TEventID outputEventIds[BUFFER_NUM] = {EVENT_ID0, EVENT_ID1};
    uint32_t bufferIndex{ 0 };
    static_assert(BUFFER_NUM * COMPUTE_LENGTH * sizeof(ElementAccumulator) * 2
        +  BUFFER_NUM * COMPUTE_LENGTH * sizeof(ElementOut) <= ArchTag::UB_SIZE, "Excedding the UB space!");
};


template<
    PaddingTag paddingTag_,
    class ArchTag_,
    class ElementIn_,
    class ElementOut_,
    class Layout_,
    uint32_t COMPUTE_LENGTH
>
struct RemovePaddingNDReduceAdd {
public:
    using ArchTag = ArchTag_;
    using ElementIn = ElementIn_;
    using ElementOut = ElementOut_;
    using Layout = Layout_;
    using CopyGm2Ub = Catlass::Epilogue::Tile::CopyGm2Ub<
        ArchTag, Gemm::GemmType<ElementIn, Catlass::layout::RowMajor>>;
    using CopyUb2Gm = Catlass::Epilogue::Tile::CopyUb2Gm<
        ArchTag, Gemm::GemmType<ElementOut, Catlass::layout::RowMajor>>;
    using ComputeLayout = Catlass::layout::RowMajor;

    using LayoutIn = Layout_;
    using LayoutOut = Layout_;

    static constexpr uint32_t ELE_NUM_PER_C0 = std::max(BYTE_PER_C0 / sizeof(ElementIn), BYTE_PER_C0 / sizeof(ElementOut));
    static constexpr PaddingTag paddingTag = paddingTag_;

    CATLASS_HOST_DEVICE static
    LayoutOut GetWorkspaceLayout(const LayoutIn& layout, uint32_t align)
    {
        if constexpr (std::is_same_v<LayoutIn, layout::RowMajor>) {
            return LayoutOut{layout.shape(0), layout.shape(1), RoundUp(layout.shape(1), align)};
        } else {
            return LayoutOut{layout.shape(0), layout.shape(1), RoundUp(layout.shape(0), align)};
        }
    }
    static size_t GetWorkspaceSize(uint32_t rows, uint32_t cols, uint32_t splitkFactor, uint32_t align = 1)
    {
        if constexpr (std::is_same_v<LayoutIn, layout::RowMajor>) {
            return static_cast<size_t>(rows) * RoundUp(cols, align) * sizeof(ElementIn) * splitkFactor;
        } else {
            return static_cast<size_t>(cols) * RoundUp(rows, align) * sizeof(ElementIn) * splitkFactor;
        }
    }

    CATLASS_DEVICE
    RemovePaddingNDReduceAdd(Arch::Resource<ArchTag> &resource)
    {
        int64_t bufferOffset = 0;
        for (uint32_t i = 0; i < BUFFER_NUM; i++) {
            inputBuffer[i] = resource.ubBuf.template GetBufferByByte<ElementIn>(bufferOffset);
            bufferOffset += COMPUTE_LENGTH * sizeof(ElementIn);
            accumulatorBuffer[i] = resource.ubBuf.template GetBufferByByte<ElementIn>(bufferOffset);
            bufferOffset += COMPUTE_LENGTH * sizeof(ElementIn);
            outputBuffer[i] = resource.ubBuf.template GetBufferByByte<ElementOut>(bufferOffset);
            bufferOffset += COMPUTE_LENGTH * sizeof(ElementOut);
        }
    }

    CATLASS_DEVICE
    ComputeLayout GetPaddingComputeLayout(layout::RowMajor const &layout)
    {
        return ComputeLayout(layout.shape(0), layout.shape(1), layout.stride(0));
    }

    CATLASS_DEVICE
    ComputeLayout GetPaddingComputeLayout(layout::ColumnMajor const &layout)
    {
        return ComputeLayout(layout.shape(1), layout.shape(0), layout.stride(1));
    }

    CopyGm2Ub copyGm2Ub;
    CopyUb2Gm copyUb2Gm;

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementOut> const &dst,
                    AscendC::GlobalTensor<ElementIn> const &src,
                    Layout const &layoutDst, Layout const &layoutSrc, uint32_t splitkFactor)
    {
        ComputeLayout computeLayoutSrc = GetPaddingComputeLayout(layoutSrc);
        ComputeLayout computeLayoutDst = GetPaddingComputeLayout(layoutDst);
        uint64_t gmSrcSliceSize = computeLayoutSrc.Capacity();

        uint32_t aivNum = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
        uint32_t aivId = AscendC::GetBlockIdx();

        // Each line is a tile.
        uint32_t tilesNum = computeLayoutSrc.shape(0);
        uint32_t tileLen = computeLayoutSrc.shape(1);
        uint32_t roundTileLen = RoundUp(computeLayoutSrc.shape(1), ELE_NUM_PER_C0);

        uint32_t tilesPerAiv = tilesNum / aivNum;
        uint32_t tileRemain = tilesNum % aivNum;
        if (aivId < tileRemain) {
            tilesPerAiv++;
        }
        uint32_t mIdx = aivId * tilesPerAiv;
        if (aivId >= tileRemain) {
            mIdx += tileRemain;
        }

        for (uint32_t i = 0; i < BUFFER_NUM; i++) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputEventIds[i]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(accumulatorEventIds[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputEventIds[i]);
        }

        uint32_t coreLoops = 0;
        if (roundTileLen > COMPUTE_LENGTH) {
            // Handle the same tile on multiple loops.
            // uint32_t loopsPerTile = (tileLen + COMPUTE_LENGTH - 1) / COMPUTE_LENGTH;
            uint32_t loopsPerTile = CeilDiv(tileLen, COMPUTE_LENGTH);
            coreLoops = tilesPerAiv * loopsPerTile;
            for (uint32_t loopIdx = 0; loopIdx < coreLoops; ++loopIdx) {
                uint32_t tileIdx = loopIdx / loopsPerTile;
                uint32_t inTileLoopIdx = loopIdx % loopsPerTile;
                uint32_t actualDataNum = COMPUTE_LENGTH;
                if (tileLen - inTileLoopIdx * COMPUTE_LENGTH < COMPUTE_LENGTH) {
                    actualDataNum = tileLen - inTileLoopIdx * COMPUTE_LENGTH;
                }

                MatrixCoord tileCoord(mIdx + tileIdx, inTileLoopIdx * COMPUTE_LENGTH);
                // tile offset in first workspace slice
                uint64_t srcTileOffset = computeLayoutSrc.GetOffset(tileCoord);

                ComputeLayout dstLayout = computeLayoutDst.GetTileLayout(MatrixCoord(1, actualDataNum));
                ComputeLayout srcLayout = computeLayoutSrc.GetTileLayout(MatrixCoord(1, actualDataNum));
                ComputeLayout &ubLayout = dstLayout;

                // copy first slice workspace tile  to accumulatorBuffer
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(accumulatorEventIds[bufferIndex]);
                copyGm2Ub(accumulatorBuffer[bufferIndex], src[srcTileOffset], ubLayout, srcLayout);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(accumulatorEventIds[bufferIndex]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(accumulatorEventIds[bufferIndex]);

                // copy and accumulate with tiles sliceIdx > 1
                for (uint32_t sliceIdx = 1; sliceIdx < splitkFactor; sliceIdx++) {
                    srcTileOffset += gmSrcSliceSize;
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputEventIds[bufferIndex]);
                    copyGm2Ub(inputBuffer[bufferIndex], src[srcTileOffset], ubLayout, srcLayout);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputEventIds[bufferIndex]);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputEventIds[bufferIndex]);
                    AscendC::Add(accumulatorBuffer[bufferIndex],
                        accumulatorBuffer[bufferIndex], inputBuffer[bufferIndex], actualDataNum);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputEventIds[bufferIndex]);
                }
                AscendC::PipeBarrier<PIPE_V>();

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputEventIds[bufferIndex]);
                if constexpr (!std::is_same_v<ElementIn, ElementOut>) {
                    if constexpr (std::is_same_v<ElementOut, half>) {
                        AscendC::Cast(outputBuffer[bufferIndex],
                            accumulatorBuffer[bufferIndex], AscendC::RoundMode::CAST_NONE, actualDataNum);
                    } else {
                        AscendC::Cast(outputBuffer[bufferIndex],
                            accumulatorBuffer[bufferIndex], AscendC::RoundMode::CAST_RINT, actualDataNum);
                    }
                } else {
                    AscendC::DataCopy(outputBuffer[bufferIndex], accumulatorBuffer[bufferIndex], actualDataNum);
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(accumulatorEventIds[bufferIndex]);

                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputEventIds[bufferIndex]);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputEventIds[bufferIndex]);
                uint64_t gmDstOffset = computeLayoutDst.GetOffset(tileCoord);
                copyUb2Gm(dst[gmDstOffset], outputBuffer[bufferIndex], dstLayout, ubLayout);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputEventIds[bufferIndex]);

                bufferIndex = (bufferIndex + 1) % BUFFER_NUM;
            }
        } else {
            // Handle multiple tile each loop.
            uint32_t tilesPerLoop = COMPUTE_LENGTH / roundTileLen;
            coreLoops = CeilDiv(tilesPerAiv, tilesPerLoop);
            for (uint32_t loopIdx = 0; loopIdx < coreLoops; ++loopIdx) {
                uint32_t tileIdx = loopIdx * tilesPerLoop;
                uint32_t actualTilesNum = tilesPerLoop;
                if (tilesPerAiv - tileIdx < tilesPerLoop) {
                    actualTilesNum = tilesPerAiv - tileIdx;
                }
                uint32_t actualDataNum = actualTilesNum * roundTileLen;

                MatrixCoord tileCoord(mIdx + tileIdx, 0);
                uint64_t srcTileOffset = computeLayoutSrc.GetOffset(tileCoord);

                ComputeLayout dstLayout = computeLayoutDst.GetTileLayout(MatrixCoord(actualTilesNum, tileLen));
                ComputeLayout srcLayout = computeLayoutSrc.GetTileLayout(MatrixCoord(actualTilesNum, tileLen));
                ComputeLayout ubLayout = ComputeLayout{actualTilesNum, tileLen, roundTileLen};

                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(accumulatorEventIds[bufferIndex]);
                copyGm2Ub(accumulatorBuffer[bufferIndex], src[srcTileOffset], ubLayout, srcLayout);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(accumulatorEventIds[bufferIndex]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(accumulatorEventIds[bufferIndex]);

                for (uint32_t sliceIdx = 1; sliceIdx < splitkFactor; sliceIdx++) {
                    srcTileOffset += gmSrcSliceSize;
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputEventIds[bufferIndex]);
                    copyGm2Ub(inputBuffer[bufferIndex], src[srcTileOffset], ubLayout, srcLayout);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputEventIds[bufferIndex]);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputEventIds[bufferIndex]);

                    AscendC::Add(accumulatorBuffer[bufferIndex],
                        accumulatorBuffer[bufferIndex], inputBuffer[bufferIndex], actualDataNum);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputEventIds[bufferIndex]);
                }
                AscendC::PipeBarrier<PIPE_V>();

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputEventIds[bufferIndex]);
                if constexpr (!std::is_same_v<ElementIn, ElementOut>) {
                    if constexpr (std::is_same_v<ElementOut, half>) {
                        AscendC::Cast(outputBuffer[bufferIndex],
                            accumulatorBuffer[bufferIndex], AscendC::RoundMode::CAST_NONE, actualDataNum);
                    } else {
                        AscendC::Cast(outputBuffer[bufferIndex],
                            accumulatorBuffer[bufferIndex], AscendC::RoundMode::CAST_RINT, actualDataNum);
                    }
                } else {
                    AscendC::DataCopy(outputBuffer[bufferIndex], accumulatorBuffer[bufferIndex], actualDataNum);
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(accumulatorEventIds[bufferIndex]);

                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputEventIds[bufferIndex]);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputEventIds[bufferIndex]);
                uint64_t gmDstOffset = computeLayoutDst.GetOffset(tileCoord);
                copyUb2Gm(dst[gmDstOffset], outputBuffer[bufferIndex], dstLayout, ubLayout);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputEventIds[bufferIndex]);

                bufferIndex = (bufferIndex + 1) % BUFFER_NUM;
            }
        }

        for (uint32_t i = 0; i < BUFFER_NUM; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputEventIds[i]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(accumulatorEventIds[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputEventIds[i]);
        }
    }

    CATLASS_DEVICE
    ~RemovePaddingNDReduceAdd() {}
private:
    static const uint32_t BUFFER_NUM = 2;
    uint32_t bufferIndex = 0;

    AscendC::LocalTensor<ElementIn> inputBuffer[BUFFER_NUM];
    AscendC::LocalTensor<ElementIn> accumulatorBuffer[BUFFER_NUM];
    AscendC::LocalTensor<ElementOut> outputBuffer[BUFFER_NUM];

    AscendC::TEventID inputEventIds[BUFFER_NUM] = {EVENT_ID0, EVENT_ID1};
    AscendC::TEventID accumulatorEventIds[BUFFER_NUM] = {EVENT_ID2, EVENT_ID3};
    AscendC::TEventID outputEventIds[BUFFER_NUM] = {EVENT_ID0, EVENT_ID1};

    static_assert(BUFFER_NUM * COMPUTE_LENGTH * (sizeof(ElementIn) * 2 + sizeof(ElementOut)) <= ArchTag::UB_SIZE,
            "Excedding the UB space!");
    static_assert(std::is_same_v<LayoutIn, layout::RowMajor> || 
        std::is_same_v<LayoutIn, layout::ColumnMajor>, "Unsported layout for RemovePaddingNDReduceAdd!");
};


// Template for Matmul kernel. Compute C = A * B
template <
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_,
    class ReduceAdd_
>
class SplitkMatmul {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;

    using BlockScheduler = BlockScheduler_;
    using ReduceAdd = ReduceAdd_;

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrC;
        LayoutC layoutC;
        GM_ADDR ptrWorkspace;
        uint32_t splitkFactor = 1;

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GemmCoord const &problemShape_, GM_ADDR ptrA_, LayoutA layoutA_, GM_ADDR ptrB_,
               LayoutB layoutB_, GM_ADDR ptrC_, LayoutC layoutC_, GM_ADDR ptrWorkspace_, uint32_t splitkFactor_)
            : problemShape(problemShape_), ptrA(ptrA_), layoutA(layoutA_), ptrB(ptrB_), layoutB(layoutB_),
              ptrC(ptrC_), layoutC(layoutC_), ptrWorkspace(ptrWorkspace_), splitkFactor(splitkFactor_) {}
    };

    struct Arguments {
        GemmCoord problemShape;
        uint32_t aicCoreNum;
        size_t workspaceElementSize;
        GM_ADDR ptrA;
        GM_ADDR ptrB;
        GM_ADDR ptrC;
    };

    static uint32_t GetSplitkFactor(uint32_t m, uint32_t n, uint32_t k, uint32_t aicCoreNum)
    {
        uint32_t maxSplitkFactor;
        if (k <= 1024) {
            // When k is less than or equal to 1024, it can be divided into at most 2 parts.
            maxSplitkFactor = 2;
        } else if (k <= 2048) {
            // When k is less than or equal to 2048, it can be divided into at most 4 parts.
            maxSplitkFactor = 4;
        } else if (k <= 4096) {
            // When k is less than or equal to 4096, it can be divided into at most 8 parts.
            maxSplitkFactor = 8;
        } else {
            // else it can be divided into at most 16 parts.
            maxSplitkFactor = 16;
        }
        uint32_t splitkFactor = 1;
        uint32_t m0 = L1TileShape::M;
        uint32_t n0 = L1TileShape::N;
        uint32_t k0 = L1TileShape::K;

        uint32_t baseTilesCount = CeilDiv(m, m0) * CeilDiv(n, n0);
        splitkFactor = std::min(aicCoreNum / baseTilesCount, maxSplitkFactor);
        // Prevent the split factor form being less than 1
        splitkFactor = std::max(splitkFactor, static_cast<uint32_t>(1));
        if (baseTilesCount < aicCoreNum) {
            while (splitkFactor + 1 <= maxSplitkFactor &&
                CeilDiv(baseTilesCount * splitkFactor, aicCoreNum) >=
                CeilDiv(baseTilesCount, aicCoreNum) * splitkFactor) {
                splitkFactor += 1;
            }
        }
        // Ensure that splitkFactor is less than the number of base tiels in the k direction.
        splitkFactor = std::min(CeilDiv(k, k0), splitkFactor);
        // If k is very large, splitting k can lead to better cache utilization.
        // If k is greater than 8192.
        if (k > 8192) {
            // split the k direction into at least 2 parts.
            splitkFactor = std::max(splitkFactor, static_cast<uint32_t>(2));
        }
        // If k is greater than 32768.
        if (k > 32768) {
            // split the k direction into at least 4 parts.
            splitkFactor = std::max(splitkFactor, static_cast<uint32_t>(4));
        }
        return splitkFactor;
    }

    static bool CanImplement(const Arguments &args)
    {
        return true;
    }

    static size_t GetWorkspaceSize(const Arguments &args)
    {
        return args.workspaceElementSize * args.problemShape.m() * args.problemShape.n() *
            GetSplitkFactor(args.problemShape.m(),
                args.problemShape.n(),
                args.problemShape.k(),
                args.aicCoreNum);
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
    {
        LayoutA layoutA{args.problemShape.m(), args.problemShape.k()};
        LayoutB layoutB{args.problemShape.k(), args.problemShape.n()};
        LayoutC layoutC{args.problemShape.m(), args.problemShape.n()};
        Params params{
            args.problemShape,
            args.ptrA,
            layoutA,
            args.ptrB,
            layoutB,
            args.ptrC,
            layoutC,
            workspace,
            GetSplitkFactor(args.problemShape.m(),
                args.problemShape.n(),
                args.problemShape.k(),
                args.aicCoreNum)};
        return params;
    }

    // Methods
    CATLASS_DEVICE
    SplitkMatmul() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    /// Executes one Matmul
    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params)
    {
        BlockScheduler matmulBlockScheduler(params.problemShape,
            GemmCoord(L1TileShape::M, L1TileShape::N, L1TileShape::K), params.splitkFactor);
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrWorkspace);

        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            // Compute block location
            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(
                blockCoord, matmulBlockScheduler.GetSplitkSliceIdx(loopIdx));

            // Compute initial location in logical coordinates
            MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
            MatrixCoord offsetB{blockCoord.k() * L1TileShape::K, blockCoord.n() * L1TileShape::N};
            MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};
            uint64_t gmOffsetA = params.layoutA.GetOffset(offsetA);
            uint64_t gmOffsetB = params.layoutB.GetOffset(offsetB);
            uint64_t gmOffsetC = params.layoutC.GetOffset(offsetC)
                + static_cast<uint64_t>(params.problemShape.m()) * static_cast<uint64_t>(params.problemShape.n())
                * static_cast<uint64_t>(matmulBlockScheduler.GetSplitkSliceIdx(loopIdx));

            // Compute block-scoped matrix multiply-add
            blockMmad(gmA[gmOffsetA], params.layoutA,
                      gmB[gmOffsetB], params.layoutB,
                      gmC[gmOffsetC], params.layoutC,
                      actualBlockShape);
        }

        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flagAicFinish);

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params)
    {
        using ElementOut = typename ReduceAdd::ElementOut;
        using ElementAccumulator = typename ReduceAdd::ElementAccumulator;

        Catlass::Arch::CrossCoreWaitFlag(flagAicFinish);
        Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();

        AscendC::GlobalTensor<ElementOut> gmC;
        AscendC::GlobalTensor<ElementAccumulator> gmWorkspace;
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementOut*>(params.ptrC));
        gmWorkspace.SetGlobalBuffer(reinterpret_cast<__gm__ ElementAccumulator*>(params.ptrWorkspace));
        ReduceAdd reduceAdd(resource);
        reduceAdd(gmC, gmWorkspace,
            static_cast<uint64_t>(params.problemShape.m()) * static_cast<uint64_t>(params.problemShape.n()),
            params.splitkFactor);

        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    static constexpr Arch::FlagID FLAG_AIC_FINISH = 0;
    Arch::CrossCoreFlag flagAicFinish{FLAG_AIC_FINISH};
    Arch::Resource<ArchTag> resource;
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_SPLITK_MATMUL_HPP