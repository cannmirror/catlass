/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// By setting the K_MAX_SHAPE_DIM macro, the dimension of the AscendC Tensor's ShapeInfo is configured to 0,
// optimizing stack space. If you need to use the ShapeInfo of the AscendC Tensor, please undefine this macro.

#include <tiling/platform/platform_ascendc.h>
#include "mla_preprocess_kernel.cpp"
#include "golden.hpp"
#include "helper.hpp"

using namespace std;

// This code section describes the parameters to execute the run function.
struct Options {
    static constexpr auto HELPER = "Usage: 34_mla_preprocess $n $he $quantMode $dtype $device\n";
    static constexpr auto MIN_ARGS = 5;

    // Define default value.
    uint32_t n{0};
    uint32_t he{0};
    QuantMode quantMode = QuantMode::PER_TENSOR_ASYMM_QUANT;
    uint32_t deviceId{0};
    string dataType = "float16";
    string dataPath = "../../examples/34_mla_preprocess/data";

    Options() = default;

    // Define function to parse the command-line arguments.
    int Parse(int argc, const char **argv)
    {
        // The number of arguments must >= 5.
        if (argc < MIN_ARGS) {
            printf(HELPER);
            return -1;
        }

        // Allocate arguments to parameters.
        n = atoi(argv[1]);
        he = atoi(argv[2]);
        quantMode = static_cast<QuantMode>(std::atoi(argv[3]));
        dataType = string(argv[4]);
        deviceId = atoi(argv[5]);

        return 0;
    }
};

static void AllocMem(uint8_t **host, uint8_t **device, size_t size) {
    ACL_CHECK(aclrtMallocHost(reinterpret_cast<void **>(host), size));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(device), size, ACL_MEM_MALLOC_HUGE_FIRST));
}

static void FreeMem(uint8_t *host, uint8_t *device) {
    ACL_CHECK(aclrtFreeHost(host));
    ACL_CHECK(aclrtFree(device));
}

void Run(const Options &options)
{
    aclrtStream stream{nullptr};

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    uint32_t aicoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    uint32_t n = options.n;
    uint32_t he = options.he;
    QuantMode quantMode = options.quantMode;
    string dataType = options.dataType;
    string dataPath = options.dataPath;

    if (dataType != "float16" && dataType != "bfloat16") {
        cerr << "[ERROR] dtype must be 'float16' or 'bfloat16'." << endl;
        return;
    }

    // data size
    uint32_t sizeHiddenState = n * he * sizeof(half);
    uint32_t sizeGamma1 = he * sizeof(half);
    uint32_t sizeQuantScale1 = sizeof(half);
    uint32_t sizeQuantOffset1 = sizeof(int8_t);
    uint32_t sizeDescale1 = he * sizeof(float);
    uint32_t outSize = n * he * sizeof(int8_t);
    uint32_t scaleSize = n * sizeof(float);

    uint8_t *hostHiddenState, *hostGamma1, *hostBeta1, *hostQuantScale1, *hostQuantOffset1, *hostDescale1, *hostTiling;
    uint8_t *deviceHiddenState, *deviceGamma1, *deviceBeta1, *deviceQuantScale1, *deviceQuantOffset1, *deviceDescale1,
        *deviceOut, *deviceScale, *deviceTiling;

    AllocMem(&hostHiddenState, &deviceHiddenState, sizeHiddenState);
    ReadFile(dataPath + "/hiddenState.bin", hostHiddenState, sizeHiddenState);
    ACL_CHECK(aclrtMemcpy(deviceHiddenState, sizeHiddenState, hostHiddenState, sizeHiddenState, ACL_MEMCPY_HOST_TO_DEVICE));

    AllocMem(&hostGamma1, &deviceGamma1, sizeGamma1);
    ReadFile(dataPath + "/gamma1.bin", hostGamma1, sizeGamma1);
    ACL_CHECK(aclrtMemcpy(deviceGamma1, sizeGamma1, hostGamma1, sizeGamma1, ACL_MEMCPY_HOST_TO_DEVICE));

    AllocMem(&hostBeta1, &deviceBeta1, sizeGamma1);
    ReadFile(dataPath + "/beta1.bin", hostBeta1, sizeGamma1);
    ACL_CHECK(aclrtMemcpy(deviceBeta1, sizeGamma1, hostBeta1, sizeGamma1, ACL_MEMCPY_HOST_TO_DEVICE));

    AllocMem(&hostQuantScale1, &deviceQuantScale1, sizeQuantScale1);
    ReadFile(dataPath + "/quantScale1.bin", hostQuantScale1, sizeQuantScale1);
    ACL_CHECK(aclrtMemcpy(deviceQuantScale1, sizeQuantScale1, hostQuantScale1, sizeQuantScale1, ACL_MEMCPY_HOST_TO_DEVICE));

    AllocMem(&hostQuantOffset1, &deviceQuantOffset1, sizeQuantOffset1);
    ReadFile(dataPath + "/quantOffset1.bin", hostQuantOffset1, sizeQuantOffset1);
    ACL_CHECK(aclrtMemcpy(deviceQuantOffset1, sizeQuantOffset1, hostQuantOffset1, sizeQuantOffset1, ACL_MEMCPY_HOST_TO_DEVICE));

    AllocMem(&hostDescale1, &deviceDescale1, sizeDescale1);
    ReadFile(dataPath + "/descale1.bin", hostDescale1, sizeDescale1);
    ACL_CHECK(aclrtMemcpy(deviceDescale1, sizeDescale1, hostDescale1, sizeDescale1, ACL_MEMCPY_HOST_TO_DEVICE));

    std::vector<int8_t> expectedOut(outSize);
    ReadFile(dataPath + "/goldenOut.bin", expectedOut.data(), outSize);
    std::vector<float> expectedScale(scaleSize);
    ReadFile(dataPath + "/goldenScale.bin", expectedScale.data(), scaleSize);

    ACL_CHECK(aclrtMalloc((void **)&deviceOut, outSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&deviceScale, scaleSize, ACL_MEM_MALLOC_HUGE_FIRST));

    // get tiling
    uint32_t sizeTiling = sizeof(MlaPreprocessTilingData);
    AllocMem(&hostTiling, &deviceTiling, sizeTiling);

    MlaPreprocessTiling::MlaPreprocessInfo mpInfo;
    mpInfo.n = n;
    mpInfo.he = he;
    MlaPreprocessTilingData mpTilingData;
    MlaPreprocessTiling::GetMpTilingParam(mpInfo, mpTilingData);
    hostTiling = reinterpret_cast<uint8_t *>(&mpTilingData);
    ACL_CHECK(aclrtMemcpy(deviceTiling, sizeTiling, hostTiling, sizeTiling, ACL_MEMCPY_HOST_TO_DEVICE));

    // Prepare FFTS address
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    if (dataType == "float16") {
        MlaPreprocessFp16<<<aicoreNum, nullptr, stream>>>(fftsAddr, deviceHiddenState, deviceGamma1, deviceBeta1, deviceQuantScale1, deviceQuantOffset1, deviceDescale1, deviceOut, deviceScale, deviceTiling);
    } else {
        MlaPreprocessBf16<<<aicoreNum, nullptr, stream>>>(fftsAddr, deviceHiddenState, deviceGamma1, deviceBeta1, deviceQuantScale1, deviceQuantOffset1, deviceDescale1, deviceOut, deviceScale, deviceTiling);
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<int8_t> hostOut(outSize);
    ACL_CHECK(aclrtMemcpy(hostOut.data(), outSize, deviceOut, outSize, ACL_MEMCPY_DEVICE_TO_HOST));
    std::vector<float> hostScale(scaleSize);
    ACL_CHECK(aclrtMemcpy(hostScale.data(), scaleSize, deviceScale, scaleSize, ACL_MEMCPY_DEVICE_TO_HOST));
    std::vector<uint64_t> outErrorIndices = golden::CompareData(hostOut, expectedOut, n * he);
    if (outErrorIndices.empty()) {
        std::cout << "Compare out success." << std::endl;
    } else {
        std::cerr << "Compare out failed. Error count: " << outErrorIndices.size() << std::endl;
    }
    std::vector<uint64_t> scaleErrorIndices = golden::CompareData(hostScale, expectedScale, n);
    if (scaleErrorIndices.empty()) {
        std::cout << "Compare scale success." << std::endl;
    } else {
        std::cerr << "Compare scale failed. Error count: " << scaleErrorIndices.size() << std::endl;
    }

    FreeMem(hostHiddenState, deviceHiddenState);
    FreeMem(hostGamma1, deviceGamma1);
    FreeMem(hostBeta1, deviceBeta1);
    FreeMem(hostQuantScale1, deviceQuantScale1);
    FreeMem(hostQuantOffset1, deviceQuantOffset1);
    FreeMem(hostDescale1, deviceDescale1);
    aclrtFree(deviceOut);
    aclrtFree(deviceScale);

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
}

int main(int argc, const char **argv)
{
    Options options;
    if (options.Parse(argc, argv) != 0) {
        return -1;
    }
    Run(options);
    return 0;
}