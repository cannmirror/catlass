/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_BLOCK_RMS_NORM_QUANT_HPP
#define CATLASS_GEMM_BLOCK_RMS_NORM_QUANT_HPP

#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/epilogue/tile/copy_gm_to_ub.hpp"
#include "catlass/epilogue/tile/copy_ub_to_gm.hpp"

namespace Catlass::Gemm::Block {

template <class ArchTag_, typename T_, bool WithBeta_, bool FastComputeMode_ = false,
          QuantMode quantMode_ = QuantMode::PER_TENSOR_ASYMM_QUANT, bool NeedDequant_ = false>
struct RmsNormAndRopeConvergence {
public:
    // 芯片类型
    using ArchTag = ArchTag_;
    // 输入数据类型
    using T = T_;
    // 其他模板参数
    using WithBeta = WithBeta_;
    using FastComputeMode = FastComputeMode_;
    using quantMode = quantMode_;
    using NeedDequant = NeedDequant_;
    // 引入需要的Tile
    using CopyGm2Ub = Catlass::Epilogue::Tile::CopyGm2Ub<ArchTag, Gemm::GemmType<int8_t, Catlass::layout::RowMajor>>;
    using CopyUb2Gm = Catlass::Epilogue::Tile::CopyUb2Gm<ArchTag, Gemm::GemmType<half, Catlass::layout::RowMajor>>;
    using CopyGm2UbFP32 = Catlass::Epilogue::Tile::CopyGm2Ub<ArchTag, Gemm::GemmType<float, Catlass::layout::RowMajor>>;
    using CopyUb2GmFP32 = Catlass::Epilogue::Tile::CopyUb2Gm<ArchTag, Gemm::GemmType<float, Catlass::layout::RowMajor>>;

    static const uint32_t BUF_FACTOR = 3;        // 1(g) + 1(sqx) + 1(sum) = 3
    static const uint32_t OFFSET_GAMMA = 0;      // the offset of gamma is 0
    static const uint32_t OFFSET_SQX = 1;        // the offset of sqx is 1
    static const uint32_t OFFSET_SUM = 2;        // the offset of sum is 2
    static const uint32_t OFFSET_ABS = 3;        // the offset of abs is 3
    static const uint32_t OFFSET_WORKSPACE = 4;  // the offset of workspace is 4
    static const uint32_t REPEAT_TIME_64 = 64;   // 64 default stride

    constexpr uint32_t MM1_OUT_SIZE = 2112;
    constexpr uint32_t SPLIT_SIZE_TWO = 1536;
    constexpr uint32_t SPLIT_RMSNRORM_SIZE_TWO = 64;
    constexpr uint32_t SPLIT_SIZE_ONE = 576;
    constexpr uint64_t BLOCK_SIZE_16 = 16;
    constexpr static uint32_t I8_C0_SIZE = 32;
    constexpr static uint32_t C0_SIZE = 16;
    constexpr uint32_t SPLIT_RMSNRORM_SIZE_ONE = 512;

    CopyGm2Ub copyGm2Ub;
    CopyUb2Gm copyUb2Gm;
    CopyGm2UbFP32 copyGm2UbFP32;
    CopyUb2GmFP32 copyUb2GmFP32;

    CATLASS_DEVICE
    RmsNormAndRopeConvergence()
    {}

    CATLASS_DEVICE
    RmsNormAndRopeConvergence(Arch::Resource<ArchTag> &resource)
    {
        // int64_t bufferOffset = 0;
        // uint32_t num_col_align_f32 = (num_col + REPEAT_TIME_64 - 1) / REPEAT_TIME_64 * REPEAT_TIME_64;
        inputTensor = resource.ubBuf.template GetBufferByByte<T>(0);
        gammaTensor = resource.ubBuf.template GetBufferByByte<T>(MM1_OUT_SIZE * 2);
        sinTensor = resource.ubBuf.template GetBufferByByte<T>(MM1_OUT_SIZE * 2 + SPLIT_RMSNRORM_SIZE_ONE * 2);
        cosTensor = resource.ubBuf.template GetBufferByByte<T>(MM1_OUT_SIZE * 2 + SPLIT_RMSNRORM_SIZE_ONE * 2 + SPLIT_RMSNRORM_SIZE_TWO * 2);
        slotMappingTensor = resource.ubBuf.template GetBufferByByte<int8_t>(MM1_OUT_SIZE * 2 + SPLIT_RMSNRORM_SIZE_ONE * 2 + SPLIT_RMSNRORM_SIZE_TWO * 4);
        rmsNormTensor = resource.ubBuf.template GetBufferByByte<float>(MM1_OUT_SIZE * 2 + SPLIT_RMSNRORM_SIZE_ONE * 2 + SPLIT_RMSNRORM_SIZE_TWO * 4 + 4096 * 32);
        tempTensor = resource.ubBuf.template GetBufferByByte<float>(MM1_OUT_SIZE * 2 + SPLIT_RMSNRORM_SIZE_ONE * 2 + SPLIT_RMSNRORM_SIZE_TWO * 4 + 4096 * 32 + SPLIT_RMSNRORM_SIZE_ONE * 3 * 4 + SPLIT_RMSNRORM_SIZE_TWO * 2 * 4 + MM1_OUT_SIZE * 4 * 2 + 32);
    }

    CATLASS_DEVICE
    void ReduceSumCustom(const AscendC::LocalTensor<float> &dst_local,
                                       const AscendC::LocalTensor<float> &src_local,
                                       const AscendC::LocalTensor<float> &work_local,
                                       int32_t count)
    {
        uint64_t mask = NUM_PER_REP_FP32;
        int32_t repeatTimes = count / NUM_PER_REP_FP32;
        int32_t tailCount = count % NUM_PER_REP_FP32;
        int32_t bodyCount = repeatTimes * NUM_PER_REP_FP32;
        AscendC::BinaryRepeatParams repeatParams;
        repeatParams.src0RepStride = AscendC::ONE_REPEAT_BYTE_SIZE / AscendC::ONE_BLK_SIZE;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1RepStride = 0;
        repeatParams.src1BlkStride = 1;
        repeatParams.dstRepStride = 0;
        repeatParams.dstBlkStride = 1;
        Duplicate(work_local, ZERO, NUM_PER_REP_FP32);
        AscendC::PipeBarrier<PIPE_V>();
        if (likely(repeatTimes > 0)) {
            Add(work_local, src_local, work_local, mask, repeatTimes, repeatParams);
            AscendC::PipeBarrier<PIPE_V>();
        }
        if (unlikely(tailCount != 0)) {
            Add(work_local, src_local[bodyCount], work_local, tailCount, 1, repeatParams);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::AscendCUtils::SetMask<float>(NUM_PER_REP_FP32);
        cadd_v<ArchType::ASCEND_V220, float>(dst_local,  // dst
                                            work_local, // src
                                            1,          // repeat
                                            0,          // dstRepeatStride
                                            1,          // srcBlockStride
                                            0);         // srcRepeatStride
        AscendC::PipeBarrier<PIPE_V>();

    }
    void operator()(
        const AscendC::LocalTensor<T1> &srcTensor, 
        
        const uint32_t sN,
        const AscendC::LocalTensor<float> &rmsNormTensor, const AscendC::LocalTensor<float> &gammaFp32,
        const AscendC::LocalTensor<float> &ropeKTensor, const AscendC::LocalTensor<float> &ropeKRevertTensor,
        const AscendC::LocalTensor<float> &calTensor, const AscendC::LocalTensor<T1> &outTmpTensor,
        AscendC::LocalTensor<half> &tmpfp16, AscendC::LocalTensor<int8_t> &int8OutTensor, float quantScale3
        
        AscendC::GlobalTensor<InDtype> gamma3GmTensor, AscendC::GlobalTensor<int32_t> slotMappingGmTensor, AscendC::GlobalTensor<int32_t> descale1gmTensor,
        const AtbOps::MlaTilingData &mlaParams_, AscendC::GlobalTensor<int32_t> s2GmTensor, AscendC::GlobalTensor<InDtype> sin1GmTensor,
        AscendC::GlobalTensor<float> s5GmTensor, AscendC::GlobalTensor<InDtype> cos1GmTensor, AscendC::GlobalTensor<K_NOPE_DTYPE> keycacheGmTensor1,
        AscendC::GlobalTensor<InDtype> keycacheGmTensor2)
    {
        this->num_col_2 = mlaParams_.rmsNumCol2;
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t sub_block_idx = static_cast<uint64_t>(GetSubBlockidx());
        uint32_t vectorBlockIdx = (blockIdx / 2) * 2 + sub_block_idx;
        int64_t slotMapGmOffset = vectorBlockIdx * row_work;
        AscendC::DataCopy(gammaTensor, gamma3GmTensor, SPLIT_RMSNRORM_SIZE_ONE); //从gamma3GmTensor连续搬运512到gammaTensor
        SET_FLAG(MTE2, V, EVENT_ID1); //vector流水线等待mte2
        WAIT_FLAG(MTE2, V, EVENT_ID1);
        Cast(gammaFp32, gammaTensor, AscendC::RoundMode::CAST_NONE, SPLIT_RMSNRORM_SIZE_ONE);
        //根据源操作数和目的操作数Tensor的数据类型进行精度转换。Cast(const LocalTensor<T>& dst, const LocalTensor<U>& src, const RoundMode& round_mode, const uint32_t count)，
        //CAST_NONE = 0, 在转换有精度损失时表示CAST_RINT模式，不涉及精度损失时表示不舍入
        //CAST_RINT,四舍六入五成双舍入
        AscendC::DataCopyPad(slotMappingTensor, slotMappingGmTensor[slotMapGmOffset],
                             AscendC::DataCopyExtParams(1, sN * sizeof(int32_t), 0, 0, 0),
                             AscendC::DataCopyPadExtParams<int32_t>(false, 0, 8 - sN % 8, 0));
        //提供数据非对齐搬运的功能，其中从Global Memory搬运数据至Local Memory时，可以根据开发者的需要自行填充数据。
        //dst src dataCopyParams padParams
        //dataCopyParams blockCount 指定该指令包含的连续传输数据块个数，数据类型为uint16_t，取值范围：blockCount∈[1, 4095]。
        //blockLen 指定该指令每个连续传输数据块长度，该指令支持非对齐搬运，每个连续传输数据块长度单位为字节。数据类型为uint16_t，blockLen不要超出该数据类型的取值范围。
        //srcStride 源操作数，相邻连续数据块的间隔
        //dstStride 目的操作数，相邻连续数据块间的间隔
        //DataCopyPadExtParams isPad false：表示用户不需要指定填充值，会默认填充随机值。
        //leftPadding连续搬运数据块左侧需要补充的数据范围，单位为元素个数。
        //rightPadding连续搬运数据块右侧需要补充的数据范围，单位为元素个数
        //paddingValue左右两侧需要填充的数据值，需要保证在数据占用字节范围内。
        if constexpr (quantMode == QuantMode::PER_TOKEN_SYMM_QUANT) {
            mmTensor = calTensor.ReinterpretCast<int32_t>()[SPLIT_SIZE_ONE];
            deScaleTensor = calTensor.ReinterpretCast<float>()[SPLIT_SIZE_ONE * 2];
            AscendC::DataCopy(deScaleTensor, descale1gmTensor, AscendC::DataCopyParams(1, SPLIT_SIZE_ONE / 8, 0, 0));
        }
        SET_FLAG(MTE2, V, EVENT_ID2);
        WAIT_FLAG(MTE2, V, EVENT_ID2);
        SET_FLAG(MTE2, S, EVENT_ID2);
        WAIT_FLAG(MTE2, S, EVENT_ID2);
        for (uint64_t loop = 0; loop < sN; ++loop) {
            uint64_t offset = vectorBlockIdx * static_cast<uint64_t>(row_work) * num_col_2 + loop * MM1_OUT_SIZE; 
            //vectorBlockIdx = (blockIdx / 2) * 2 + sub_block_idx;  blockIdx = AscendC::GetBlockIdx();获取当前核的index，用于代码内部的多核逻辑控制及多核偏移量计算等。  
            //sub_block_idx = static_cast<uint64_t>(GetSubBlockidx());获取AI Core上Vector核的ID。
            //this->num_col_2 = mlaParams_.rmsNumCol2;
            //row_work = (num_row + num_core_ - 1) / num_core_;
            //MM1_OUT_SIZE = 2112
            int64_t slotValue = static_cast<int64_t>(slotMappingTensor.GetValue(loop));
            if (slotValue == -1) {
                continue;
            }
            if constexpr (quantMode == QuantMode::PER_TENSOR_ASYMM_QUANT) {
                AscendC::DataCopy(srcTensor, s3GmTensor[offset],
                                  AscendC::DataCopyParams(1, MM1_OUT_SIZE / BLOCK_SIZE_16, 0, 0));
            } else {
                // quantMode == QuantMode::PER_TOKEN_SYMM_QUANT
                AscendC::DataCopy(mmTensor, s2GmTensor[offset], AscendC::DataCopyParams(1, SPLIT_SIZE_ONE / 8, 0, 0));
            }
            AscendC::DataCopy(sinTensor, sin1GmTensor[(row_work * vectorBlockIdx + loop) * SPLIT_RMSNRORM_SIZE_TWO],
                              SPLIT_RMSNRORM_SIZE_TWO);
            AscendC::DataCopy(cosTensor, cos1GmTensor[(row_work * vectorBlockIdx + loop) * SPLIT_RMSNRORM_SIZE_TWO],
                              SPLIT_RMSNRORM_SIZE_TWO);
            SET_FLAG(MTE2, V, EVENT_ID0);
            // ND
            uint64_t cacheStart = static_cast<uint64_t>(slotValue) * static_cast<uint64_t>(SPLIT_SIZE_ONE);
            uint64_t cacheStart1 = static_cast<uint64_t>(slotValue) * static_cast<uint64_t>(SPLIT_RMSNRORM_SIZE_ONE);
            uint64_t cacheStart2 = static_cast<uint64_t>(slotValue) * static_cast<uint64_t>(SPLIT_RMSNRORM_SIZE_TWO);
            // NZ
            uint32_t outer_idx = slotValue / 128;
            uint32_t inner_idx = slotValue % 128;

            SET_FLAG(S, MTE3, EVENT_ID0);
            /* RmsNorm start */
            WAIT_FLAG(MTE2, V, EVENT_ID0);
            if constexpr (quantMode == QuantMode::PER_TOKEN_SYMM_QUANT) {
                /* DeQuant */
                AscendC::Cast(mmTensor.ReinterpretCast<float>(), mmTensor, AscendC::RoundMode::CAST_NONE,
                              SPLIT_SIZE_ONE);
                AscendC::PipeBarrier<PIPE_V>();
                // 阻塞相同流水，具有数据依赖的相同流水之间需要插此同步。
                AscendC::Mul(mmTensor.ReinterpretCast<float>(), mmTensor.ReinterpretCast<float>(), deScaleTensor,
                             SPLIT_SIZE_ONE);
                //dst src0、src1 count参与计算的元素个数
                AscendC::PipeBarrier<PIPE_V>();
                float perTokenDescale = s5GmTensor.GetValue(row_work * vectorBlockIdx + loop);
                SET_FLAG(S, V, EVENT_ID0);
                WAIT_FLAG(S, V, EVENT_ID0);
                AscendC::Muls(mmTensor.ReinterpretCast<float>(), mmTensor.ReinterpretCast<float>(), perTokenDescale,
                              SPLIT_SIZE_ONE);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(srcTensor, mmTensor.ReinterpretCast<float>(), AscendC::RoundMode::CAST_RINT,
                              SPLIT_SIZE_ONE);
                //根据源操作数和目的操作数Tensor的数据类型进行精度转换。Cast(const LocalTensor<T>& dst, const LocalTensor<U>& src, const RoundMode& round_mode, const uint32_t count)，
                //CAST_RINT,四舍六入五成双舍入
                AscendC::PipeBarrier<PIPE_V>();
            }
            Cast(rmsNormTensor, srcTensor, AscendC::RoundMode::CAST_NONE, SPLIT_RMSNRORM_SIZE_ONE);
            AscendC::PipeBarrier<PIPE_V>();
            Mul(calTensor, rmsNormTensor, rmsNormTensor, SPLIT_RMSNRORM_SIZE_ONE);
            AscendC::PipeBarrier<PIPE_V>();
            ReduceSumCustom(calTensor[SPLIT_RMSNRORM_SIZE_ONE], calTensor, calTensor[SPLIT_RMSNRORM_SIZE_ONE * 2],
                            SPLIT_RMSNRORM_SIZE_ONE);
            //向量化求和 dst_local src_local(要被加起来的原始数据) work_local(临时计算 暂存结果) count(要加多少个数)
            SET_FLAG(V, S, EVENT_ID1);
            WAIT_FLAG(V, S, EVENT_ID1);
            float rms = sqrt(calTensor.GetValue(SPLIT_RMSNRORM_SIZE_ONE) / SPLIT_RMSNRORM_SIZE_ONE + epsilon_);
            SET_FLAG(S, V, EVENT_ID1);
            WAIT_FLAG(S, V, EVENT_ID1);
            AscendC::PipeBarrier<PIPE_V>();
            Duplicate(calTensor, rms, SPLIT_RMSNRORM_SIZE_ONE);
            //将一个变量或一个立即数，复制多次并填充到向量。
            //dst scalarValue被复制的源操作数 count参与计算的元素个数。
            AscendC::PipeBarrier<PIPE_V>();
            Div(calTensor, rmsNormTensor, calTensor, SPLIT_RMSNRORM_SIZE_ONE);
            //dst src0 src1(dst = src0 / src1) count参与计算的元素个数
            AscendC::PipeBarrier<PIPE_V>();
            Mul(rmsNormTensor, gammaFp32, calTensor, SPLIT_RMSNRORM_SIZE_ONE);

            AscendC::PipeBarrier<PIPE_V>();
            if constexpr (CACHE_MODE == CACHE_MODE_INT8_NZCACHE) {
                // quant
                Muls(rmsNormTensor, rmsNormTensor, quantScale3, SPLIT_RMSNRORM_SIZE_ONE);
                AscendC::PipeBarrier<PIPE_V>();
                CastFrom32To16(tmpfp16, rmsNormTensor, SPLIT_RMSNRORM_SIZE_ONE);
                AscendC::PipeBarrier<PIPE_V>();
                CastFromF16ToI8(int8OutTensor, tmpfp16, -128, SPLIT_RMSNRORM_SIZE_ONE);
                AscendC::PipeBarrier<PIPE_V>();
            } else {
                AscendC::PipeBarrier<PIPE_V>();
                if (std::is_same<T1, __bf16>::value) {
                    Cast(outTmpTensor, rmsNormTensor, AscendC::RoundMode::CAST_RINT, SPLIT_RMSNRORM_SIZE_ONE);
                } else {
                    Cast(outTmpTensor, rmsNormTensor, AscendC::RoundMode::CAST_NONE, SPLIT_RMSNRORM_SIZE_ONE);
                }
            }
            /* RmsNorm end */
            /* Rope K start */
            uint64_t revertOffset = SPLIT_RMSNRORM_SIZE_TWO / 2;
            Cast(ropeKTensor, srcTensor[SPLIT_RMSNRORM_SIZE_ONE], AscendC::RoundMode::CAST_NONE,
                 SPLIT_RMSNRORM_SIZE_TWO);
            Cast(ropeKRevertTensor[revertOffset], srcTensor[SPLIT_RMSNRORM_SIZE_ONE], AscendC::RoundMode::CAST_NONE,
                 revertOffset);
            Cast(ropeKRevertTensor, srcTensor[SPLIT_RMSNRORM_SIZE_ONE + revertOffset], AscendC::RoundMode::CAST_NONE,
                 revertOffset);
            Duplicate(calTensor, static_cast<float>(-1), revertOffset);
            Duplicate(calTensor[revertOffset], static_cast<float>(1), revertOffset);
            AscendC::PipeBarrier<PIPE_V>();
            Cast(calTensor[SPLIT_RMSNRORM_SIZE_TWO], cosTensor, AscendC::RoundMode::CAST_NONE, SPLIT_RMSNRORM_SIZE_TWO);
            Cast(calTensor[SPLIT_RMSNRORM_SIZE_TWO * 2], sinTensor, AscendC::RoundMode::CAST_NONE,
                 SPLIT_RMSNRORM_SIZE_TWO);
            AscendC::PipeBarrier<PIPE_V>();
            Mul(ropeKTensor, calTensor[SPLIT_RMSNRORM_SIZE_TWO], ropeKTensor, SPLIT_RMSNRORM_SIZE_TWO);
            Mul(ropeKRevertTensor, calTensor[SPLIT_RMSNRORM_SIZE_TWO * 2], ropeKRevertTensor, SPLIT_RMSNRORM_SIZE_TWO);
            AscendC::PipeBarrier<PIPE_V>();
            Mul(ropeKRevertTensor, calTensor, ropeKRevertTensor, SPLIT_RMSNRORM_SIZE_TWO);
            AscendC::PipeBarrier<PIPE_V>();
            Add(ropeKRevertTensor, ropeKTensor, ropeKRevertTensor, SPLIT_RMSNRORM_SIZE_TWO);
            AscendC::PipeBarrier<PIPE_V>();
            if (std::is_same<T1, __bf16>::value) {
                Cast(outTmpTensor[SPLIT_RMSNRORM_SIZE_ONE], ropeKRevertTensor, AscendC::RoundMode::CAST_RINT,
                     SPLIT_RMSNRORM_SIZE_TWO);
            } else {
                Cast(outTmpTensor[SPLIT_RMSNRORM_SIZE_ONE], ropeKRevertTensor, AscendC::RoundMode::CAST_NONE,
                     SPLIT_RMSNRORM_SIZE_TWO);
            }
            AscendC::PipeBarrier<PIPE_V>();
            /* Rope K end */
            SET_FLAG(V, MTE3, EVENT_ID0);
            WAIT_FLAG(V, MTE3, EVENT_ID0);
            WAIT_FLAG(S, MTE3, EVENT_ID0);
            if constexpr (CACHE_MODE == CACHE_MODE_KVCACHE) {
                DataCopy(keycacheGmTensor1[cacheStart], outTmpTensor, SPLIT_SIZE_ONE);
            } else if constexpr (CACHE_MODE == CACHE_MODE_INT8_NZCACHE) {
                uint64_t cacheSatartI8Nz1 = outer_idx * 128 * 512 + inner_idx * I8_C0_SIZE;
                uint64_t cacheSatartNz2 = outer_idx * 128 * 64 + inner_idx * C0_SIZE;
                // nope:int8 nz
                AscendC::DataCopyExtParams outExt;
                outExt.blockCount = SPLIT_RMSNRORM_SIZE_ONE / I8_C0_SIZE;
                //指定该指令包含的连续传输数据块个数，数据类型为uint16_t，取值范围：blockCount∈[1, 4095]。
                outExt.blockLen = I8_C0_SIZE * sizeof(int8_t);
                //指定该指令每个连续传输数据块长度
                outExt.srcStride = 0;
                //源操作数，相邻连续数据块的间隔（前面一个数据块的尾与后面数据块的头的间隔），如果源操作数的逻辑位置为VECIN/VECOUT，则单位为dataBlock(32Bytes), 如果源操作数的逻辑位置为GM,则单位为Byte。
                outExt.dstStride = (128 * I8_C0_SIZE - I8_C0_SIZE) * sizeof(int8_t);
                //目的操作数，相邻连续数据块间的间隔（前面一个数据块的尾与后面数据块的头的间隔），如果目的操作数的逻辑位置为VECIN/VECOUT，则单位为dataBlock(32Bytes)，如果目的操作数的逻辑位置为GM，则单位为Byte。
                DataCopyPad(keycacheGmTensor1[cacheSatartI8Nz1], int8OutTensor, outExt);
                //提供数据非对齐搬运的功能，其中从Global Memory搬运数据至Local Memory时，可以根据开发者的需要自行填充数据。
                //dst src dataCopyParams padParams
                // rope:T1 nz
                outExt.blockCount = SPLIT_RMSNRORM_SIZE_TWO / C0_SIZE;
                outExt.blockLen = C0_SIZE * sizeof(T1);
                outExt.srcStride = 0;
                outExt.dstStride = (128 * C0_SIZE - C0_SIZE) * sizeof(T1);
                DataCopyPad(keycacheGmTensor2[cacheSatartNz2], outTmpTensor[SPLIT_RMSNRORM_SIZE_ONE], outExt);
            } else if constexpr (CACHE_MODE == CACHE_MODE_NZCACHE) {
                uint64_t cacheSatartNz1 = outer_idx * 128 * 512 + inner_idx * C0_SIZE;
                uint64_t cacheSatartNz2 = outer_idx * 128 * 64 + inner_idx * C0_SIZE;
                // nope:T1 nz
                AscendC::DataCopyExtParams outExt;
                outExt.blockCount = SPLIT_RMSNRORM_SIZE_ONE / C0_SIZE;
                outExt.blockLen = C0_SIZE * sizeof(T1);
                outExt.srcStride = 0;
                outExt.dstStride = (128 * C0_SIZE - C0_SIZE) * sizeof(T1);
                DataCopyPad(keycacheGmTensor1[cacheSatartNz1], outTmpTensor, outExt);
                // rope:T1 nz
                outExt.blockCount = SPLIT_RMSNRORM_SIZE_TWO / C0_SIZE;
                outExt.blockLen = C0_SIZE * sizeof(T1);
                outExt.srcStride = 0;
                outExt.dstStride = (128 * C0_SIZE - C0_SIZE) * sizeof(T1);
                DataCopyPad(keycacheGmTensor2[cacheSatartNz2], outTmpTensor[SPLIT_RMSNRORM_SIZE_ONE], outExt);
            } else {
                // keycache1
                DataCopy(keycacheGmTensor1[cacheStart1], outTmpTensor, SPLIT_RMSNRORM_SIZE_ONE);
                // keycache2
                DataCopy(keycacheGmTensor2[cacheStart2], outTmpTensor[SPLIT_RMSNRORM_SIZE_ONE],
                         SPLIT_RMSNRORM_SIZE_TWO);
            }
            SET_FLAG(MTE3, MTE2, EVENT_ID1);
            WAIT_FLAG(MTE3, MTE2, EVENT_ID1);
        }
    }

private:
    AscendC::LocalTensor<InDtype> inputTensor
    AscendC::LocalTensor<InDtype> sinTensor
    AscendC::LocalTensor<InDtype> cosTensor
    AscendC::LocalTensor<int32_t> slotMappingTensor
    AscendC::LocalTensor<float> tmp32Tensor
    AscendC::LocalTensor<InDtype> tempTensor



    AscendC::LocalTensor<int8_t> dstTensor;
    AscendC::LocalTensor<T> srcTensor;
    AscendC::LocalTensor<T> gammaTensor;
    AscendC::LocalTensor<T> betaTensor;
    AscendC::LocalTensor<T> quantScaleTensor;
    AscendC::LocalTensor<int8_t> quantOffsetTensor;
    AscendC::LocalTensor<float> res1Tensor;
    AscendC::LocalTensor<float> res3Tensor;
    AscendC::LocalTensor<float> fp32_xy;
    AscendC::LocalTensor<float> buf;
    AscendC::LocalTensor<int32_t> mmTensor;
    AscendC::LocalTensor<float> deScaleTensor;

    AscendC::GlobalTensor<T> gammaGmTensor;
    AscendC::GlobalTensor<T> betaGmTensor;
    AscendC::GlobalTensor<T> quantScaleGmTensor;
    AscendC::GlobalTensor<int8_t> quantOffsetGmTensor;
    AscendC::GlobalTensor<T> inputGmTensor;
    AscendC::GlobalTensor<int8_t> outputGmTensor;
    AscendC::GlobalTensor<float> perTokenDescaleGmTensor;
    AscendC::GlobalTensor<float> perChannelDescaleGmTensor;
    AscendC::GlobalTensor<int32_t> mmGmTensor;

    uint32_t num_col_{0};       // 输入的列数
    uint32_t num_row_{0};       // 输入的行数
    uint32_t row_work_{0};      // 需要计算多少行
    uint32_t row_work{0};       // 需要计算多少行
    uint32_t row_step_{0};      // 除最后一次，每次搬入多少行
    uint32_t row_tail_{0};      // 最后一次搬入多少行数据
    uint64_t gm_offset_{0};     // GM数据起始位置偏移量
    uint64_t gm_out_offset_{0}; // GM数据起始位置偏移量
    float avg_factor_{1.0};     // num_col_的倒数
    float input_scale_{1.0};    // 非对称量化系数
    float input_offset_{0};     // 非对称量化偏移适配高精度
    int32_t input_stride_{0};
    float epsilon_{1e-12f}; // norm平滑参数
    uint32_t num_col_align_int8{0};
    uint32_t num_col_align_f16{0};
    uint32_t num_col_align_f32{0};
    uint32_t num_col_align_f32_long{0};
    uint32_t num_col_align_withStride_int8{0};
    uint32_t num_col_align_withStride_fp16{0};
    uint32_t num_col_align_withStride_fp32{0};
    uint32_t num_col_temp;
    half quantMin_{-128};
    uint32_t num_slice_{0};
    uint32_t tail_size_{0};
    uint32_t tail_copy_{0};

    uint32_t num_col_2;
};

}  // namespace Catlass::Gemm::Block

#endif  // CATLASS_GEMM_BLOCK_RMS_NORM_QUANT_HPP