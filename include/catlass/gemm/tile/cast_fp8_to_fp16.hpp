/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_TILE_CAST_FP8_TO_FP16_HPP
#define CATLASS_GEMM_TILE_CAST_FP8_TO_FP16_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/helper.hpp"

namespace Catlass::Gemm::Tile {

template <
    class ArchTag,
    class SrcType_,
    class DstType_,
    uint32_t COMPUTE_LENGTH
>
struct TileCastFp8ToFp16Dequant {
    using ElementSrc = typename SrcType_::Element;
    using ElementDst = typename DstType_::Element;
    using LayoutSrc = typename SrcType_::Layout;
    using LayoutDst = typename DstType_::Layout;

    static const uint32_t Alignment = 256;
    static constexpr uint32_t ELE_NUM_PER_BLK = BYTE_PER_BLK / sizeof(int8_t);

    struct Params {
        half scalar;
        half zeroPoint;

        CATLASS_HOST_DEVICE
        Params() = default;

        CATLASS_HOST_DEVICE
        Params(half scalar_, half zeroPoint_)
        {
            scalar = scalar_;
            zeroPoint = zeroPoint_;
        }
    }

    CATLASS_DEVICE
    TileCastFp8ToFp16Dequant()
    {}

    CATLASS_DEVICE
    TileCastFp8ToFp16Dequant(Arch::Resource<ArchTag> &resource, Params const &params_) : params(params_)
    {
        int64_t bufferOffset = 0;
        for (uint32_t i = 0; i < BUFFER_NUM; i++) {
            inputBuffer[i] = resource.ubBuf.template GetBufferByByte<ElementIn>(bufferOffset * sizeof(ElementIn));
            bufferOffset += COMPUTE_LENGTH;
        }
        for (uint32_t i = 0; i < BUFFER_NUM; i++) {
            outputBuffer[i] = (resource.ubBuf.template GetBufferByByte<ElementIn>(bufferOffset * sizeof(ElementIn)))
                                  .template ReinterpretCast<half>();
            bufferOffset += COMPUTE_LENGTH * 2;
        }
        for (uint32_t i = 0; i < BUFFER_NUM; i++) {
            workspace[i] = (resource.ubBuf.template GetBufferByByte<ElementIn>(bufferOffset * sizeof(ElementIn)))
                               .template ReinterpretCast<half>();
            bufferOffset += COMPUTE_LENGTH * 2;
        }
        int16_t value_uint = 0x4000;
        value_vector1 = (resource.ubBuf.template GetBufferByByte<ElementIn>(bufferOffset * sizeof(ElementIn)))
                            .template ReinterpretCast<int16_t>();
        bufferOffset += 256;
        AscendC::Duplicate<int16_t>(value_vector1, value_uint, 128);
        AscendC::PipeBarrier<PIPE_V>();
        // AscendC::PipeBarrier<PIPE_V>();
        value_uint = 0x3FFF;
        value_vector2 = (resource.ubBuf.template GetBufferByByte<ElementIn>(bufferOffset * sizeof(ElementIn)))
                            .template ReinterpretCast<int16_t>();
        bufferOffset += 256;
        AscendC::Duplicate<int16_t>(value_vector2, value_uint, 128);
        AscendC::PipeBarrier<PIPE_V>();
        // AscendC::PipeBarrier<PIPE_V>();
    }

    CATLASS_DEVICE
    void Dequant(AscendC::LocalTensor<int8_t> &src, AscendC::LocalTensor<half> &dst,
        AscendC::LocalTensor<int16_t> &value_vector1, AscendC::LocalTensor<int16_t> &value_vector2,
        AscendC::LocalTensor<half> &workspace, half scalar, half zeroPoint)
    {
        pipe_barrier(PIPE_V);
        uint32_t num = COMPUTE_LENGTH;
        num = (num + 128 - 1) / 128 * 128;
        AscendC::Cast<half, uint8_t>(dst.template ReinterpretCast<half>(),
            src.template ReinterpretCast<uint8_t>(),
            AscendC::RoundMode::CAST_NONE,
            num);
        pipe_barrier(PIPE_V);

        AscendC::Adds<half>(dst, dst, 1024, num);
        pipe_barrier(PIPE_V);

        AscendC::ShiftLeft<uint16_t>(
            dst.template ReinterpretCast<uint16_t>(), dst.template ReinterpretCast<uint16_t>(), 7, num);
        pipe_barrier(PIPE_V);

        uint64_t mask = 128;
        AscendC::And<int16_t>(workspace.template ReinterpretCast<int16_t>(),
            dst.template ReinterpretCast<int16_t>(),
            value_vector1,
            mask,
            num / 128,
            {1, 1, 1, 8, 8, 0});
        pipe_barrier(PIPE_V);

        AscendC::ShiftLeft<uint16_t>(
            workspace.template ReinterpretCast<uint16_t>(), workspace.template ReinterpretCast<uint16_t>(), 1, num);
        pipe_barrier(PIPE_V);

        AscendC::And<int16_t>(dst.template ReinterpretCast<int16_t>(),
            dst.template ReinterpretCast<int16_t>(),
            value_vector2,
            mask,
            num / 128,
            {1, 1, 1, 8, 8, 0});
        pipe_barrier(PIPE_V);

        AscendC::Or<int16_t>(dst.template ReinterpretCast<int16_t>(),
            dst.template ReinterpretCast<int16_t>(),
            workspace.template ReinterpretCast<int16_t>(),
            num);
        pipe_barrier(PIPE_V);

        AscendC::Muls<half>(dst.template ReinterpretCast<half>(), dst.template ReinterpretCast<half>(), 1 << 8, num);
        pipe_barrier(PIPE_V);

        AscendC::Adds(dst, dst, params.zeroPoint, num);
        pipe_barrier(PIPE_V);

        AscendC::Muls(dst, dst, params.scalar, num);
        pipe_barrier(PIPE_V);
    }

    CATLASS_DEVICE
    void operator() (
        AscendC::GlobalTensor<ElementDst> gmDst, LayoutDst const &layoutDst,
        AscendC::GlobalTensor<ElementSrc> gmSrc, LayoutSrc const &layoutSrc,
        uint32_t srcStride, uint32_t dstStride, uint32_t &bufferIndex
    )
    {
        static_assert(std::is_same_v(LayoutSrc, layout::ColumnMajor) || 
                    std::is_same_v(LayoutSrc, layout::RowMajor), 
                    "The Layout must be Row/Column major given to `TileCastFp8ToFp16Dequant`");
        uint32_t tilesNum, tileLen;
        if constexpr (std::is_same_v(LayoutSrc, layout::RowMajor)) {
            tilesNum = layoutSrc.shape(0);
            tileLen = layoutSrc.shape(1);
        } else if constexpr (std::is_same_v(LayoutSrc, layout::ColumnMajor)) {
            tilesNum = layoutSrc.shape(1);
            tileLen = layoutSrc.shape(0);
        }

        uint32_t tilesPerAiv = tilesNum / AscendC::GetSubBlockNum();
        uint32_t tilesRemain = tilesNum % AscendC::GetSubBlockNum();
        if (AscendC::GetSubBlockIdx() < tilesRemain) {
            tilesPerAiv++;
        }

        uint64_t taskSrcOffset = AscendC::GetSubBlockIdx() * tilesPerAiv * srcStride;
        uint64_t taskDstOffset = AscendC::GetSubBlockIdx() * tilesPerAiv * dstStride;
        if (AscendC::GetSubBlockIdx() >= tilesRemain) {
            taskSrcOffset += tilesRemain * srcStride;
            taskDstOffset += tilesRemain * dstStride;
        }

        uint32_t totalLoops = 0;
        uint32_t loopsPerTile, tilesInALoop;
        uint32_t tileLenRoundFp8 = RoundUp<Alignment, uint32_t>(tileLen);
        if (tileLenRoundFp8 > COMPUTE_LENGTH / 2) { 
            // One signle tile length is bigger than COMPUTE_LENGTH, which should be clipped.
            loopsPerTile = CeilDiv(tileLen, COMPUTE_LENGTH);
            totalLoops = taskPerAiv * loopsPerTile;
        }else if (tileLenRoundFp8 != 0) { 
            // COMPUTE_LENGTH is bigger than tile length, such that more than one tiles can be arranged together.
            tilesInALoop = COMPUTE_LENGTH / tileLenRoundFp8;
            totalLoops = CeilDiv(taskPerAiv, tilesInALoop);
        } // tileLenRoundFp8 == 0 --> totalLoops = 0
        
        uint32_t tileTailLen = tileLen % COMPUTE_LENGTH;
        uint64_t srcProcessOffset, dstProcessOffset;
        uint32_t loadLen = COMPUTE_LENGTH, storeLen, loadRepeat = 1, storeRepeat = 1;
        // uint32_t srcLoadStride = srcStride, dstLoadStride = tileLenRoundFp8, 
        //         srcStoreStride = tileLenRoundFp8, dstStoreStride = dstStride;
        for (int ldx=0; ldx < totalLoops; ldx++) {

            // Dynamic compute length
            if (tileLenRoundFp8 > COMPUTE_LENGTH / 2) {
                uint32_t fullTileRounds    = ldx / loopsPerTile;
                uint32_t residueTileRounds = ldx % loopsPerTile;
                srcProcessOffset = taskSrcOffset + fullTileRounds * srcStride + residueTileRounds * COMPUTE_LENGTH;
                dstProcessOffset = taskDstOffset + fullTileRounds * dstStride + residueTileRounds * COMPUTE_LENGTH;
                
                if ((residueTileRounds == loopsPerTile - 1) && (tileTailLen != 0)) {
                    loadLen = tileTailLen;
                }
            }else {
                uint32_t fullTileRounds = ldx * tilesInALoop;
                srcProcessOffset = taskSrcOffset + fullTileRounds * srcStride;
                dstProcessOffset = taskDstOffset + fullTileRounds * dstStride;

                loadLen = tileLen;
                storeLen = tileLen;
                loadRepeat = tilesInALoop;
                storeRepeat = tilesInALoop;

                if ((ldx == totalLoops - 1) && (taskPerAiv % tilesInALoop != 0)) {
                    loadRepeat = taskPerAiv % tilesInALoop;
                    storeRepeat = loadRepeat;
                }
            }
            // uint32_t srcLoadStride = srcStride;
            // uint32_t dstLoadStride = tileLenRoundFp8;
            // uint32_t srcStoreStride = tileLenRoundFp8;
            
            // GM -> UB
            AscendC::DataCopyExtParams dataCopyParamsIn(
                loadRepeat,
                loadLen * sizeof(ElementSrc),
                (srcStride - loadLen) * sizeof(ElementSrc), // 
                (tileLenRoundFp8 - loadLen) * sizeof(ElementSrc) / BYTE_PER_BLK
            );
            AscendC::DataCopyPadExtParams<ElementSrc> padParams(false, 0, 0, 0);

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EventIdBuffer[bufferIndex]);
            AscendC::DataCopyPad(inputBuffer[bufferIndex], gmSrc[srcProcessOffset], 
                                    dataCopyParamsIn, padParams);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EventIdBuffer[bufferIndex]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EventIdBuffer[bufferIndex]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EventIdBuffer[bufferIndex]);
            
            Dequant(inputBuffer[bufferIndex],
                outputBuffer[bufferIndex],
                value_vector1,
                value_vector2,
                workspace[bufferIndex]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EventIdBuffer[bufferIndex]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EventIdBuffer[bufferIndex]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EventIdBuffer[bufferIndex]);

            AscendC::DataCopyExtParams dataCopyParams(
                storeRepeat,
                storeLen * sizeof(ElementDst),
                (tileLenRoundFp8 - storeLen) * sizeof(ElementDst) / BYTE_PER_C0,
                (dstStride - storeLen) * sizeof(ElementDst),
                0
            );
            AscendC::DataCopyPad(gmDst[dstProcessOffset], outputBuffer[bufferIndex], dataCopyParams);
        }

        uint32_t tilesNum = layoutSrc.shape(0);
        uint32_t tiles
    }

private:
    static const uint32_t BUFFER_NUM = 2;
    AscendC::LocalTensor<int8_t> inputBuffer[BUFFER_NUM];
    AscendC::LocalTensor<int16_t> value_vector1;
    AscendC::LocalTensor<int16_t> value_vector2;
    AscendC::LocalTensor<half> outputBuffer[BUFFER_NUM];
    AscendC::LocalTensor<half> workspace[BUFFER_NUM];
    AscendC::TEventID EventIdBuffer[BUFFER_NUM] = {EVENT_ID0, EVENT_ID1};
    AscendC::TEventID EventIdBufferForCast[BUFFER_NUM] = {EVENT_ID2, EVENT_ID3};
    uint32_t bufferIndexForCast{0};

    Params params;
};

} // namespace Catlass::Gemm::Tile

#endif // CATLASS_GEMM_TILE_CAST_FP8_TO_FP16_HPP