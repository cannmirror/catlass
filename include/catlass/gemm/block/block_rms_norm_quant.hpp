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
struct RmsNormQuant {
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

    CopyGm2Ub copyGm2Ub;
    CopyUb2Gm copyUb2Gm;
    CopyGm2UbFP32 copyGm2UbFP32;
    CopyUb2GmFP32 copyUb2GmFP32;

    CATLASS_DEVICE
    RmsNormQuant()
    {}

    CATLASS_DEVICE
    RmsNormQuant(Arch::Resource<ArchTag> &resource, uint32_t num_col)
    {
        // int64_t bufferOffset = 0;
        uint32_t num_col_align_f32 = (num_col + REPEAT_TIME_64 - 1) / REPEAT_TIME_64 * REPEAT_TIME_64;
        srcTensor = resource.ubBuf.template GetBufferByByte<T>(0);
        gammaTensor = resource.ubBuf.template GetBufferByByte<T>(HIDDTEN_STATE * 2);
        betaTensor = resource.ubBuf.template GetBufferByByte<T>(HIDDTEN_STATE * 2 + HIDDTEN_STATE * 2);
        quantScaleTensor = resource.ubBuf.template GetBufferByByte<T>(HIDDTEN_STATE * 2 + HIDDTEN_STATE * 2 + HIDDTEN_STATE * 2);
        quantOffsetTensor = resource.ubBuf.template GetBufferByByte<int8_t>(HIDDTEN_STATE * 2 + HIDDTEN_STATE * 2 + HIDDTEN_STATE * 2 + 32);
        res1Tensor = resource.ubBuf.template GetBufferByByte<float>(HIDDTEN_STATE * 2 + HIDDTEN_STATE * 2 + HIDDTEN_STATE * 2 + 64);
        res3Tensor = resource.ubBuf.template GetBufferByByte<float>(HIDDTEN_STATE * 2 + HIDDTEN_STATE * 2 + HIDDTEN_STATE * 2 + 64 + num_col_align_f32 * 4);
        dstTensor = resource.ubBuf.template GetBufferByByte<int8_t>(HIDDTEN_STATE * 2 + HIDDTEN_STATE * 2 + HIDDTEN_STATE * 2 + 64 + num_col_align_f32 * 4 + BUF_FACTOR * num_col_align_f32 * 4 + 64);
    }

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<T> &gammaGmTensor, AscendC::GlobalTensor<T> &betaGmTensor,
                                AscendC::GlobalTensor<T> &quantScaleGmTensor,
                                AscendC::GlobalTensor<int8_t> &quantOffsetGmTensor, GM_ADDR perTokenDescaleGm,
                                GM_ADDR perChannelDescaleGm, GM_ADDR gmInput, GM_ADDR gmOutput, uint32_t stride,
                                uint32_t num_col, float avg_factor, uint64_t gm_offset, uint64_t gm_out_offset,
                                uint32_t row_work_, uint32_t n)
    {
        this->gammaGmTensor = gammaGmTensor;
        this->betaGmTensor = betaGmTensor;
        this->quantScaleGmTensor = quantScaleGmTensor;
        this->quantOffsetGmTensor = quantOffsetGmTensor;
        this->perTokenDescaleGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(perTokenDescaleGm));
        this->perChannelDescaleGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(perChannelDescaleGm));
        if constexpr (!NEED_DEQUANT) {
            inputGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(gmInput));
        } else {
            mmGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(gmInput));
        }
        outputGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(gmOutput));

        num_col_ = num_col;
        avg_factor_ = avg_factor;
        epsilon_ = 1e-6;
        quantMin_ = -128;
        this->num_row_ = n;
        this->row_work = row_work;
        this->row_work_ = row_work_;
        gm_offset_ = gm_offset;
        gm_out_offset_ = gm_out_offset;
        num_col_align_int8 = (num_col_ + REPEAT_TIME_256 - 1) / REPEAT_TIME_256 * REPEAT_TIME_256;
        num_col_align_f16 = (num_col_ + REPEAT_TIME_128 - 1) / REPEAT_TIME_128 * REPEAT_TIME_128;
        num_col_align_f32 = (num_col_ + REPEAT_TIME_64 - 1) / REPEAT_TIME_64 * REPEAT_TIME_64;
        input_stride_ = stride;

        num_col_align_withStride_int8 =
            (num_col_ - input_stride_ + REPEAT_TIME_256 - 1) / REPEAT_TIME_256 * REPEAT_TIME_256;
        num_col_align_withStride_fp16 =
            (num_col_ - input_stride_ + REPEAT_TIME_128 - 1) / REPEAT_TIME_128 * REPEAT_TIME_128;
        num_col_align_withStride_fp32 =
            (num_col_ - input_stride_ + REPEAT_TIME_64 - 1) / REPEAT_TIME_64 * REPEAT_TIME_64;
        
        fp32_xy = res1Tensor;
        buf = res3Tensor;
        AscendC::LocalTensor<float> g = buf[OFFSET_GAMMA * num_col_align_withStride_fp32];           // 0
        AscendC::LocalTensor<float> sqx = buf[OFFSET_SQX * num_col_align_withStride_fp32];           // 1
        AscendC::LocalTensor<float> work = buf[OFFSET_SUM * num_col_align_withStride_fp32];          // 2
        AscendC::LocalTensor<float> abs = buf[OFFSET_ABS * num_col_align_withStride_fp32];           // 3
        AscendC::LocalTensor<float> sum = buf[OFFSET_WORKSPACE * num_col_align_withStride_fp32];     // 4
        AscendC::LocalTensor<float> max = buf[OFFSET_WORKSPACE * num_col_align_withStride_fp32 + 8]; // 5
        AscendC::LocalTensor<float> perTokenDescaleTensor =
            buf[OFFSET_WORKSPACE * num_col_align_withStride_fp32 + 16]; // 6

        AscendC::DataCopy(gammaTensor, gammaGmTensor,
                          AscendC::DataCopyParams(1, (num_col_ - input_stride_) / BLOCK_SIZE_16, 0, 0));
        AscendC::DataCopy(betaTensor, betaGmTensor,
                          AscendC::DataCopyParams(1, (num_col_ - input_stride_) / BLOCK_SIZE_16, 0, 0));
        SET_FLAG(MTE2, V, EVENT_ID1);
        if constexpr (quantMode == QuantMode::PER_TENSOR_ASYMM_QUANT) {
            AscendC::DataCopy(quantScaleTensor, quantScaleGmTensor, AscendC::DataCopyParams(1, 1, 0, 0));
            AscendC::DataCopy(quantOffsetTensor, quantOffsetGmTensor, AscendC::DataCopyParams(1, 1, 0, 0));
        }

        if constexpr (NEED_DEQUANT) {
            mmTensor = buf.ReinterpretCast<int32_t>()[OFFSET_WORKSPACE * num_col_align_withStride_fp32 + 16];
            deScaleTensor = buf[OFFSET_WORKSPACE * num_col_align_withStride_fp32 + 16 + MM1_OUT_SIZE];
            perTokenDescaleTensor = buf[OFFSET_WORKSPACE * num_col_align_withStride_fp32 + 16 + MM1_OUT_SIZE * 2];
            AscendC::DataCopy(deScaleTensor, perChannelDescaleGmTensor, AscendC::DataCopyParams(1, num_col_ / 8, 0, 0));
        }

        if constexpr (quantMode == QuantMode::PER_TENSOR_ASYMM_QUANT) {
            if (std::is_same<T, __bf16>::value) {
                SET_FLAG(MTE2, V, EVENT_ID0);
                WAIT_FLAG(MTE2, V, EVENT_ID0);
                Cast(g, quantScaleTensor, AscendC::RoundMode::CAST_NONE, 1);
                AscendC::SetFlag<HardEvent::V_S>(EVENT_ID0);
                AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID0);
                input_scale_ = 1 / (float)(g.GetValue(0));
                input_offset_ = (float)(quantOffsetTensor.GetValue(0));
            } else {
                SET_FLAG(MTE2, S, EVENT_ID0);
                WAIT_FLAG(MTE2, S, EVENT_ID0);
                input_scale_ = 1 / (float)(quantScaleTensor.GetValue(0));
                input_offset_ = (float)(quantOffsetTensor.GetValue(0));
            }
            AscendC::SetFlag<HardEvent::S_V>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::S_V>(EVENT_ID0);
        }
        WAIT_FLAG(MTE2, V, EVENT_ID1);
        Cast(buf[OFFSET_GAMMA * num_col_align_withStride_fp32], gammaTensor, AscendC::RoundMode::CAST_NONE,
             REPEAT_TIME_64, num_col_align_withStride_fp32 / REPEAT_TIME_64,
             {1, 1, AscendC::DEFAULT_REPEAT_STRIDE, AscendC::DEFAULT_REPEAT_STRIDE / OFFSET_SUM});
        AscendC::PipeBarrier<PIPE_V>();
        uint64_t pid = 0;
        SET_FLAG(MTE3, MTE2, EVENT_ID0);
        while (pid < row_work_) {
            uint64_t offset = pid * num_col_;
            uint64_t outOffset = pid * (num_col_ - input_stride_);
            WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
            if constexpr (!NEED_DEQUANT) {
                AscendC::DataCopy(srcTensor, inputGmTensor[gm_offset_ + offset],
                                  AscendC::DataCopyParams(1, num_col_ / BLOCK_SIZE_16, 0, 0));
                SET_FLAG(MTE2, V, EVENT_ID0);
                WAIT_FLAG(MTE2, V, EVENT_ID0);
            } else {
                /* Dequant start */
                AscendC::DataCopy(mmTensor, mmGmTensor[gm_offset_ + offset],
                                  AscendC::DataCopyParams(1, num_col_ / 8, 0, 0)); // 2112
                SET_FLAG(MTE2, V, EVENT_ID0);
                WAIT_FLAG(MTE2, V, EVENT_ID0);
                AscendC::Cast(mmTensor.ReinterpretCast<float>(), mmTensor, AscendC::RoundMode::CAST_NONE, num_col_);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(mmTensor.ReinterpretCast<float>(), mmTensor.ReinterpretCast<float>(), deScaleTensor,
                             num_col_);
                SET_FLAG(V, MTE2, EVENT_ID0);
                WAIT_FLAG(V, MTE2, EVENT_ID0);
                gm_to_ub_align<ArchType::ASCEND_V220, float>(perTokenDescaleTensor, perTokenDescaleGmTensor[pid],
                                                             0,             // sid
                                                             1,             // nBurst
                                                             sizeof(float), // lenBurst
                                                             0,             // leftPaddingNum
                                                             0,             // rightPaddingNum
                                                             0,             // srcGap
                                                             0              // dstGap
                );
                SET_FLAG(MTE2, S, EVENT_ID0);
                WAIT_FLAG(MTE2, S, EVENT_ID0);
                float perTokenDescale = perTokenDescaleTensor.GetValue(0);
                SET_FLAG(S, V, EVENT_ID0);
                WAIT_FLAG(S, V, EVENT_ID0);
                AscendC::Muls(mmTensor.ReinterpretCast<float>(), mmTensor.ReinterpretCast<float>(), perTokenDescale,
                              num_col_);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(srcTensor, mmTensor.ReinterpretCast<float>(), AscendC::RoundMode::CAST_RINT, num_col_);
                AscendC::PipeBarrier<PIPE_V>();
            }

            Cast(fp32_xy, srcTensor[input_stride_], AscendC::RoundMode::CAST_NONE, REPEAT_TIME_64,
                 num_col_align_withStride_fp32 / REPEAT_TIME_64,
                 {1, 1, AscendC::DEFAULT_REPEAT_STRIDE, AscendC::DEFAULT_REPEAT_STRIDE / OFFSET_SUM});
            AscendC::PipeBarrier<PIPE_V>();
            Mul(sqx, fp32_xy, fp32_xy, REPEAT_TIME_64, num_col_align_withStride_fp32 / REPEAT_TIME_64,
                {1, 1, 1, AscendC::DEFAULT_REPEAT_STRIDE, AscendC::DEFAULT_REPEAT_STRIDE,
                 AscendC::DEFAULT_REPEAT_STRIDE});
            AscendC::PipeBarrier<PIPE_V>();
            Muls(sqx, sqx, avg_factor_, num_col_ - input_stride_);
            AscendC::PipeBarrier<PIPE_V>();
            ReduceSumCustom(sum, sqx, work, num_col_ - input_stride_);
            AscendC::PipeBarrier<PIPE_V>();
            Adds(sum, sum, epsilon_, 1);
            AscendC::PipeBarrier<PIPE_V>();
            Sqrt(sum, sum, 1);
            SET_FLAG(V, S, EVENT_ID0);
            WAIT_FLAG(V, S, EVENT_ID0);
            float factor = 1 / sum.GetValue(0);
            SET_FLAG(S, V, EVENT_ID0);
            WAIT_FLAG(S, V, EVENT_ID0);
            Muls(fp32_xy, fp32_xy, factor, REPEAT_TIME_64, num_col_align_withStride_fp32 / REPEAT_TIME_64,
                 {1, 1, AscendC::DEFAULT_REPEAT_STRIDE, AscendC::DEFAULT_REPEAT_STRIDE});
            AscendC::PipeBarrier<PIPE_V>();
            Mul(fp32_xy, fp32_xy, g, REPEAT_TIME_64, num_col_align_withStride_fp32 / REPEAT_TIME_64,
                {1, 1, 1, AscendC::DEFAULT_REPEAT_STRIDE, AscendC::DEFAULT_REPEAT_STRIDE,
                 AscendC::DEFAULT_REPEAT_STRIDE});
            AscendC::PipeBarrier<PIPE_V>();
            if constexpr (WITH_BETA) {
                AscendC::LocalTensor<T> b = this->betaTensor;
                Cast(work, b, AscendC::RoundMode::CAST_NONE, REPEAT_TIME_64,
                     num_col_align_withStride_fp32 / REPEAT_TIME_64,
                     {1, 1, AscendC::DEFAULT_REPEAT_STRIDE, AscendC::DEFAULT_REPEAT_STRIDE / OFFSET_SUM});
                AscendC::PipeBarrier<PIPE_V>();
                Add(fp32_xy, fp32_xy, work, REPEAT_TIME_64, num_col_align_withStride_fp32 / REPEAT_TIME_64,
                    {1, 1, 1, AscendC::DEFAULT_REPEAT_STRIDE, AscendC::DEFAULT_REPEAT_STRIDE,
                     AscendC::DEFAULT_REPEAT_STRIDE});
                AscendC::PipeBarrier<PIPE_V>();
            }
            /* Quant start */
            if constexpr (quantMode == QuantMode::PER_TENSOR_ASYMM_QUANT) {
                Muls(fp32_xy, fp32_xy, input_scale_, REPEAT_TIME_64, num_col_align_withStride_fp32 / REPEAT_TIME_64,
                     {1, 1, AscendC::DEFAULT_REPEAT_STRIDE, AscendC::DEFAULT_REPEAT_STRIDE});
                AscendC::PipeBarrier<PIPE_V>();
                Adds(fp32_xy, fp32_xy, input_offset_, REPEAT_TIME_64, num_col_align_withStride_fp32 / REPEAT_TIME_64,
                     {1, 1, AscendC::DEFAULT_REPEAT_STRIDE, AscendC::DEFAULT_REPEAT_STRIDE});
                AscendC::PipeBarrier<PIPE_V>();
            } else if constexpr (quantMode == QuantMode::PER_TOKEN_SYMM_QUANT) {
                Abs(abs, fp32_xy, REPEAT_TIME_64, num_col_align_withStride_fp32 / REPEAT_TIME_64,
                    {1, 1, AscendC::DEFAULT_REPEAT_STRIDE, AscendC::DEFAULT_REPEAT_STRIDE});
                AscendC::PipeBarrier<PIPE_V>();
                ReduceMax(max, abs, work, num_col_ - input_stride_);
                AscendC::PipeBarrier<PIPE_V>();
                float scaleOut = max.GetValue(0) / 127;
                SET_FLAG(S, V, EVENT_ID0);
                WAIT_FLAG(S, V, EVENT_ID0);
                Muls(fp32_xy, fp32_xy, (float)(1 / scaleOut), REPEAT_TIME_64,
                     num_col_align_withStride_fp32 / REPEAT_TIME_64,
                     {1, 1, AscendC::DEFAULT_REPEAT_STRIDE, AscendC::DEFAULT_REPEAT_STRIDE});
                AscendC::PipeBarrier<PIPE_V>();
                perTokenDescaleTensor.SetValue(0, scaleOut);
                SET_FLAG(S, MTE3, EVENT_ID0);
                WAIT_FLAG(S, MTE3, EVENT_ID0);
                if constexpr (!NEED_DEQUANT) {
                    ub_to_gm_align<ArchType::ASCEND_V220, float>(perTokenDescaleGmTensor[pid], perTokenDescaleTensor, 0,
                                                                 1,                 // nBurst
                                                                 1 * sizeof(float), // lenBurst
                                                                 0,                 // leftPaddingNum
                                                                 0,                 // rightPaddingNum
                                                                 0,                 // srcGap
                                                                 0                  // dstGap
                    );
                } else {
                    ub_to_gm_align<ArchType::ASCEND_V220, float>(perTokenDescaleGmTensor[num_row_ + pid],
                                                                 perTokenDescaleTensor, 0,
                                                                 1,                 // nBurst
                                                                 1 * sizeof(float), // lenBurst
                                                                 0,                 // leftPaddingNum
                                                                 0,                 // rightPaddingNum
                                                                 0,                 // srcGap
                                                                 0                  // dstGap
                    );
                }
                SET_FLAG(MTE3, V, EVENT_ID0);
                WAIT_FLAG(MTE3, V, EVENT_ID0);
            }

            AscendC::LocalTensor<half> tmpfp16 =
                buf.ReinterpretCast<half>()[OFFSET_SUM * num_col_align_withStride_fp32 * 2];
            CastFrom32To16(tmpfp16, fp32_xy, num_col_align_withStride_fp32);
            AscendC::PipeBarrier<PIPE_V>();
            CastFromF16ToI8(dstTensor, tmpfp16, quantMin_, num_col_align_withStride_fp16);
            AscendC::PipeBarrier<PIPE_V>();
            SET_FLAG(V, MTE3, EVENT_ID0);
            WAIT_FLAG(V, MTE3, EVENT_ID0);
            AscendC::DataCopy(outputGmTensor[gm_out_offset_ + outOffset], dstTensor,
                              AscendC::DataCopyParams(1, (num_col_ - input_stride_) / 32, 0, 0));
            SET_FLAG(MTE3, MTE2, EVENT_ID0);
            ++pid;
        }
        WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
    }

private:
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
};

}  // namespace Catlass::Gemm::Block

#endif  // CATLASS_GEMM_BLOCK_RMS_NORM_QUANT_HPP