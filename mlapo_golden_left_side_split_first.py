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
            if slot < 0:
                continue
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

    def RmsNormAndRopeAndReshapeAndCacheGolden(self, x, gamma, keyRope, cos, sin, slotMapping, keycachein):
        rmsNormOutput = self.rmsNormGolden(x, gamma)
        ropeOutput = self.RopeGolden(keyRope, sin, cos)
        ropeReshape = ropeOutput.reshape(self.input_token_num, 1, self.rope_hidden_size)

        keyRAC = torch.cat((rmsNormOutput, ropeReshape), axis=-1)
        return self.RACGolden(keyRAC, slotMapping, keycachein)

    def calc_vec_mm_data(self, N, headNum, data_type, quant_mode):
        if quant_mode == 0:
            mm1In = self.rms_norm_quant_calc(self.input1, self.gamma1, self.beta1, self.quantScale1, self.quantOffset1)
        else:
            [mm1In, perTokenDescale1] = self.rmsNormPerTokenGolden([self.input1, self.gamma1, self.beta1])

        self.rmsquantOut1 = mm1In.clone()
        mm1Out = torch.matmul(mm1In.to(torch.float32), self.wdqkv.transpose(0, 1).to(torch.float32))

        mm1OutSplit1, mm1OutSplit2 = torch.split(mm1Out, [576, 1536], dim=1)

        bias_11, bias_12 = torch.split(self.bias1, [576, 1536], dim=1)  # [1, 576], [1, 1536]
        deScale11 = self.deScale1[:576]      # [576]
        deScale12 = self.deScale1[576:]    # [1536]

        if quant_mode == 0:
            mm1OutSplit1 = mm1OutSplit1.to(torch.int32) + bias_11
            mm1OutSplit1 = (mm1OutSplit1.to(torch.float32) * deScale11).to(data_type)
        else:
            perTokenDescale1 = perTokenDescale1.unsqueeze(1).expand(-1, 576)
            mm1OutSplit1 = (mm1OutSplit1.to(torch.float32) * deScale11 * perTokenDescale1).to(data_type)
        # self.mm1Out1 = mm1Out
        # if data_type == torch.float16 and quant_mode == 0:
        #     self.deScale1 = process_deq_scale(deq_scale=self.deScale1)
        mm11OutSplit1, mm12OutSplit1 = torch.split(mm1OutSplit1, [512, 64], dim=1) #切分成512 64两部分
        mm11OutSplit1 = mm11OutSplit1.reshape(N, 1, 512)
        self.keyOutTensor = self.keyCache.clone()
        self.keyOut1 = self.RmsNormAndRopeAndReshapeAndCacheGolden(
            mm11OutSplit1, self.gamma3, mm12OutSplit1, self.cos1, self.sin1, self.slotMapping, self.keyCache
        )#左边通路两个输出
        print(self.keyOut1.shape)
        result1 = self.keyOut1[..., 0:512]
        result2 = self.keyOut1[..., 512:576]
        print(result1.shape)
        print(result2.shape)

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

        self.gamma3 = torch.rand(size=(512,)).to(data_type)
        self.sin1 = torch.rand(size=(N, 64)).to(data_type)
        self.cos1 = torch.rand(size=(N, 64)).to(data_type)
        self.keyCache = torch.rand(size=(blockNum, blockSize, 1, headdim)).to(data_type)

        self.slotMapping = torch.from_numpy(np.random.choice(192 * 128, N, replace=False).astype(np.int32)).to(
            torch.int32
        )
        self.slotMapping[0] = -1

        self.bias1 = torch.from_numpy(np.random.randint(-10, 10, (1, 2112)).astype(np.int32)).to(torch.int32)

        self.beta1 = torch.from_numpy(np.random.randint(-2, 2, (hiddenStrate)).astype(np.float16)).to(data_type)

        self.calc_vec_mm_data(N, headNum, data_type, quant_mode)

if __name__ == "__main__":
    num_tokens = 32#32
    num_heads = 32#32
    quant_mode = 0
    cache_mode = 1
    data_type = torch.float16
    test = test_kvcache()
    test.calc_vec_mm_atb_data(num_tokens, num_heads, data_type, cache_mode, quant_mode)





