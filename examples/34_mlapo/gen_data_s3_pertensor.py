#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import unittest
import numpy as np
import torch
import torch_npu
import torch.nn.functional as F
import sys,os

import random
import logging



OP_NAME = "MlaPreprocessOperation"
QUANTMAX = 127
QUANTMIN = -128
block_size = 128

random.seed(12)
np.random.seed(12)
torch.manual_seed(12)


def process_deq_scale(deq_scale: torch.Tensor) -> np.ndarray:
    ret = torch.frombuffer(deq_scale.numpy().tobytes(), dtype=torch.int32).to(torch.int64)
    return ret


class test_kvcache:

    def rmsNormPerTokenGolden(self, intensors):
        out_shape = intensors[0].shape
        input0 = intensors[0].float()
        input1 = intensors[1].float()
        input2 = intensors[2].float()
        square_sum = torch.sum(torch.square(input0), axis=-1, keepdims=True)
        factor = 1.0 / torch.sqrt(square_sum / out_shape[-1] + self.epsilon)
        output = input0 * factor * input1
        if torch.numel(input2) != 0:
            output = output + input2
        output = output.half()
        scale, _ = torch.max(torch.abs(output), dim=-1, keepdim=True)
        scale = torch.div(torch.tensor([127], dtype=torch.float32), scale)
        output = torch.mul(output, scale)
        out_scale = torch.div(torch.tensor([1], dtype=torch.float32), scale)
        output = torch.clamp(torch.round(output.float()), min=QUANTMIN, max=QUANTMAX)
        return [output.to(torch.int8), out_scale.squeeze(-1).to(torch.float32)]

    def rms_norm_quant_calc(
        self,
        input: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        quantScale: torch.Tensor,
        quantOffset: torch.Tensor,
    ):
        out_shape = input.shape
        scale = 1.0 / quantScale.float().item()
        offset = quantOffset.float().item()
        square_sum = torch.sum(torch.square(input.float()), axis=-1, keepdims=True)
        factor = 1.0 / torch.sqrt(square_sum / out_shape[-1] + self.epsilon)
        output = input.float() * factor * gamma.float()
        output = (output + beta.float()) * scale + offset
        output = torch.round(output).half()
        output = torch.min(output, torch.tensor(QUANTMAX, dtype=torch.half))
        output = torch.max(output, torch.tensor(QUANTMIN, dtype=torch.half)).to(torch.int8)
        return output

    def RACGolden(self, keyRAC, slotMapping, keycacheout_golden):
        for i, slot in enumerate(slotMapping):
            # if slot < 0:
            #     continue
            block_index = slot // block_size
            block_offset = slot % block_size
            token_key = keyRAC[i]

            keycacheout_golden[block_index][block_offset] = token_key
        return keycacheout_golden

    def RopeGolden(self, keyRope, sin, cos):
        RopeGolden = keyRope * cos + self.rotateHalf(keyRope) * sin
        return RopeGolden

    def rotateHalf(self, k_temp):
        first_half, second_half = torch.chunk(k_temp, 2, dim=1)
        processed_k_split = torch.cat((-second_half, first_half), dim=1)
        return processed_k_split

    def rmsNormGolden(self, x, gamma):
        x_float32 = x.to(torch.float32)
        square_sum = torch.sum(torch.square(x_float32), axis=-1, keepdims=True)
        rms = 1.0 / torch.sqrt(square_sum / self.rms_hidden_size + self.epsilon)
        gamma_float32 = gamma.to(torch.float32)
        rmsNorm = rms * x_float32 * gamma_float32
        result = rmsNorm.to(self.dtype)
        return result

    def s8_saturation(self, inputdata):
        inputdata = torch.min(inputdata, torch.tensor(QUANTMAX, dtype=torch.float16))
        inputdata = torch.max(inputdata, torch.tensor(QUANTMIN, dtype=torch.float16))
        return np.rint(inputdata).to(torch.int8)

    def quant(self,x, qscale):
        # qscale = qscale.to(torch.float)
        qscale = 1 / qscale
        x = x.to(torch.float)
        # 使用广播机制来避免显式的循环
        scaled_values = (x * qscale).to(torch.float16)
        s8_res_cal = self.s8_saturation(scaled_values)
        return s8_res_cal

    def RmsNormAndRopeAndReshapeAndCacheGolden(self, x, gamma, keyRope, cos, sin, slotMapping, keycachein):
        self.rmsNormOutput = self.rmsNormGolden(x, gamma)
        self.rmsNormOutputQuant = self.quant(self.rmsNormOutput, self.quantScale3)
        ropeOutput = self.RopeGolden(keyRope, sin, cos)
        ropeReshape = ropeOutput.reshape(self.input_token_num, 1, self.rope_hidden_size)

        keyRAC = torch.cat((self.rmsNormOutputQuant, ropeReshape), axis=-1)
        return self.RACGolden(keyRAC, slotMapping, keycachein)

    def calc_vec_mm_data(self, N, headNum, data_type, quant_mode):
        if quant_mode == 0:
            mm1In = self.rms_norm_quant_calc(self.input1, self.gamma1, self.beta1, self.quantScale1, self.quantOffset1)
        else:
            [mm1In, perTokenDescale1] = self.rmsNormPerTokenGolden([self.input1, self.gamma1, self.beta1])

        self.rmsquantOut1 = mm1In.clone()
        mm1Out = torch.matmul(mm1In.to(torch.float32), self.wdqkv.transpose(0, 1).to(torch.float32))
        if quant_mode == 0:
            mm1Out = mm1Out.to(torch.int32) + self.bias1
            mm1Out = (mm1Out.to(torch.float32) * self.deScale1).to(data_type)
        else:
            perTokenDescale1 = perTokenDescale1.unsqueeze(1).expand(-1, 2112)
            mm1Out = (mm1Out.to(torch.float32) * self.deScale1 * perTokenDescale1).to(data_type)
        self.mm1Out1 = mm1Out
        if data_type == torch.float16 and quant_mode == 0:
            self.deScale1 = process_deq_scale(deq_scale=self.deScale1)
        mm1OutSplit1, mm1OutSplit2 = torch.split(mm1Out, [576, 1536], dim=1)
        print("s3_shape={}, dtype={}".format(mm1OutSplit1.shape, mm1OutSplit1.dtype))
        print(mm1OutSplit1)
        mm1OutSplit1.numpy().tofile("data/s3_pertensor.bin")

        mm11OutSplit1, mm12OutSplit1 = torch.split(mm1OutSplit1, [512, 64], dim=1) #切分成512 64两部分
        mm11OutSplit1 = mm11OutSplit1.reshape(N, 1, 512)
        self.keyOutTensor = self.keyCache.clone()
        self.keyOut1 = self.RmsNormAndRopeAndReshapeAndCacheGolden(
            mm11OutSplit1, self.gamma3, mm12OutSplit1, self.cos1, self.sin1, self.slotMapping, self.keyCache
        )#左边通路两个输出
        result1 = self.keyOut1[..., 0:512]
        result1.numpy().tofile("data/keycache1_pertensor.bin")
        result2 = self.keyOut1[..., 512:576]
        result2.numpy().tofile("data/keycache2_pertensor.bin")
        print("keycache1_shape={}, dtype={}".format(result1.shape, result1.dtype))
        print("keycache2_shape={}, dtype={}".format(result2.shape, result2.dtype))
        print(self.keyOut1.shape)

    def calc_vec_mm_atb_data(self, N, headNum, data_type, cacheMode, quant_mode=0):
        hiddenStrate = 7168#7168
        blockNum = 192#192
        blockSize = 128#128
        headdim = 576#576
        self.input_token_num = N
        self.rms_hidden_size = 512#512
        self.rope_hidden_size = 64#64
        self.headNum = headNum
        self.epsilon = 1e-6#1e-6
        self.dtype = data_type

        self.input1 = torch.from_numpy(np.random.uniform(-2.0, 2.0, size=(N, 7168))).to(data_type)  
        self.gamma1 = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(hiddenStrate))).to(data_type)
        self.quantScale1 = torch.from_numpy(np.random.uniform(-2.0, 2.0, size=(1))).to(data_type)
        self.quantOffset1 = torch.from_numpy(np.random.uniform(-128.0, 127.0, size=(1))).to(torch.int8)
        self.wdqkv = torch.from_numpy(np.random.uniform(-2.0, 2.0, size=(2112, 7168))).to(torch.int8)  

        self.deScale1 = torch.rand((2112), dtype=torch.float32) / 1000
        self.deScale1.numpy().tofile("data/deScale1_pertensor.bin")

        self.gamma3 = torch.rand(size=(512,)).to(data_type)
        # print(self.gamma3)
        self.gamma3.numpy().tofile("data/gamma3_pertensor.bin")

        self.sin1 = torch.rand(size=(N, 64)).to(data_type)
        self.cos1 = torch.rand(size=(N, 64)).to(data_type)
        self.cos1.numpy().tofile("data/cos1_pertensor.bin")
        self.sin1.numpy().tofile("data/sin1_pertensor.bin")
        print("cos1_shape={}, dtype={}".format(self.cos1.shape, self.cos1.dtype))
        print("sin1_shape={}, dtype={}".format(self.sin1.shape, self.sin1.dtype))

        self.quantScale3 = torch.from_numpy(np.random.uniform(-2.0, 2.0, size=(1))).to(data_type)
        self.quantScale3.numpy().tofile("data/quantScale3_pertensor.bin")
        print("quantScale3_shape={}, dtype={}".format(self.quantScale3.shape, self.quantScale3.dtype))
        print("self.quantScale3:{}".format(self.quantScale3))

        self.keyCache = torch.rand(size=(blockNum, blockSize, 1, headdim)).to(data_type)

        self.slotMapping = torch.from_numpy(np.random.choice(192 * 128, N, replace=False).astype(np.int32)).to(
            torch.int32
        )
        # self.slotMapping[0] = -1
        self.slotMapping.numpy().tofile("data/slotMapping_pertensor.bin")

        self.bias1 = torch.from_numpy(np.random.randint(-10, 10, (1, 2112)).astype(np.int32)).to(torch.int32)
        self.bias1.numpy().tofile("data/bias1_pertensor.bin")

        self.beta1 = torch.from_numpy(np.random.randint(-2, 2, (hiddenStrate)).astype(np.float16)).to(data_type)
        self.beta1.numpy().tofile("data/beta1_pertensor.bin")

        self.calc_vec_mm_data(N, headNum, data_type, quant_mode)

    def process_from_576_to_keycache(
        self,
        input_576: torch.Tensor,          # [N, 576]
        gamma3: torch.Tensor,             # [512]
        quantScale3: torch.Tensor,        # scalar, float16
        sin: torch.Tensor,                # [N, 64]
        cos: torch.Tensor,                # [N, 64]
        slotMapping: torch.Tensor,        # [N], int32
        keyCache: torch.Tensor,           # [blockNum, blockSize, 1, 576], dtype=float16
        epsilon: float = 1e-6,
    ):
        assert input_576.shape[-1] == 576
        N = input_576.shape[0]
        device = input_576.device
        dtype = input_576.dtype

        # Split input
        x_rms = input_576[:, :512]      # [N, 512]
        x_rope = input_576[:, 512:]     # [N, 64]

        # ---- RMSNorm + Quant (per-tensor) ----
        x_f32 = x_rms.to(torch.float32)
        square_sum = torch.sum(x_f32 * x_f32, dim=-1, keepdim=True)  # [N, 1]
        rms = torch.rsqrt(square_sum / 512.0 + epsilon)              # [N, 1]
        normed = x_f32 * rms * gamma3.to(torch.float32)              # [N, 512]
        normed = normed.to(dtype)                                    # back to float16
        print(f"normed:{normed}")

        # Quantize to int8 using per-tensor scale
        scale = quantScale3.item()                                   # scalar
        inv_scale = 1.0 / scale
        scaled = normed * inv_scale                                  # [N, 512]
        # Round and clamp to int8 range
        quantized = torch.clamp(torch.round(scaled), min=QUANTMIN, max=QUANTMAX).to(torch.int8)
        print(f"quantized:{quantized}")

        # ---- RoPE on last 64 dims ----
        def rotate_half(x):
            x1, x2 = torch.chunk(x, 2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)

        rope_out = x_rope * cos + rotate_half(x_rope) * sin          # [N, 64]
        rope_out = rope_out.to(dtype)

        # ---- Concat and reshape for RAC ----
        combined = torch.cat([quantized, rope_out], dim=-1)          # [N, 576]
        combined = combined.unsqueeze(1)                             # [N, 1, 576]

        # ---- Write into keyCache via slotMapping ----
        keyCache_out = keyCache.clone()
        block_size = 128
        for i, slot in enumerate(slotMapping):
            block_idx = slot // block_size
            offset = slot % block_size
            keyCache_out[block_idx, offset, 0, :] = combined[i, 0, :]

        # Split output
        keycache1 = keyCache_out[..., :512]   # [blockNum, blockSize, 1, 512]
        print(f"keycache1:{keycache1[..., 0:16]}")
        keycache2 = keyCache_out[..., 512:]   # [blockNum, blockSize, 1, 64]
        print(f"keycache2:{keycache2}")
        return keycache1, keycache2

if __name__ == "__main__":
    num_tokens = 1#32
    num_heads = 32#32
    quant_mode = 0#pertensor
    cache_mode = 1
    data_type = torch.float16
    test = test_kvcache()
    test.calc_vec_mm_atb_data(num_tokens, num_heads, data_type, cache_mode, quant_mode)
    #------------------------------------------------------------------------------------------
#     N = num_tokens
#     blockNum = 192#192
#     blockSize = 128#128
#     headdim = 576
#     input576_ones = torch.from_numpy(np.ones((N, 576))).to(data_type)
#     input576_ones.numpy().tofile("data/input576_ones_pertensor.bin")

#     gamma3_ones = torch.from_numpy(np.ones((512,))).to(data_type)
#     gamma3_ones.numpy().tofile("data/gamma3_ones_pertensor.bin")

#     quantScale3_ones = torch.from_numpy(np.ones((1))).to(data_type)
#     quantScale3_ones.numpy().tofile("data/quantScale3_ones_pertensor.bin")

#     sin_zeros = torch.from_numpy(np.zeros((N, 64))).to(data_type)
#     sin_zeros.numpy().tofile("data/sin_zeros_pertensor.bin")

#     cos_ones = torch.from_numpy(np.ones((N, 64))).to(data_type)
#     cos_ones.numpy().tofile("data/cos_ones_pertensor.bin")

#     slotMapping_r = torch.from_numpy(np.random.choice(192 * 128, N, replace=False).astype(np.int32)).to(
#             torch.int32
#         )
#     print(f"slot:{slotMapping_r}")
#     slotMapping_r.numpy().tofile("data/slotMapping_r_pertensor.bin")

#     keyCache_r = torch.rand(size=(blockNum, blockSize, 1, headdim)).to(data_type)

#     keycache1_ones, keycache2_ones = test.process_from_576_to_keycache(
#     input576_ones,
#     gamma3_ones,
#     quantScale3_ones,
#     sin_zeros,
#     cos_ones,
#     slotMapping_r,
#     keyCache_r,
#     0
# )
#     keycache1_ones.numpy().tofile("data/keycache1_ones_pertensor.bin")
#     keycache2_ones.numpy().tofile("data/keycache2_ones_pertensor.bin")






