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
    static constexpr auto HELPER = "Usage: 34_mla_preprocess $rmsNumCol2 $n $quantMode $dtype $device\n";
    static constexpr auto MIN_ARGS = 5;

    // Define default value.
    uint32_t rmsNumCol2{0};
    uint32_t n{0};

    QuantMode quantMode = QuantMode::PER_TENSOR_ASYMM_QUANT;
    uint32_t deviceId{0};
    string dataType = "float16";
    string dataPath = "/home/chenyuning/1103catlass/catlass/examples/34_mlapo/data";

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
        rmsNumCol2 = atoi(argv[1]);
        n = atoi(argv[2]);
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

    uint32_t rmsNumCol2 = options.rmsNumCol2;

    QuantMode quantMode = options.quantMode;
    string dataType = options.dataType;
    string dataPath = options.dataPath;

    if (dataType != "float16" && dataType != "bfloat16") {
        cerr << "[ERROR] dtype must be 'float16' or 'bfloat16'." << endl;
        return;
    }

    // data size
    uint32_t sizeGamma3 = 512 * sizeof(half);
    uint32_t sizeSlotMapping1 = 32 * sizeof(int32_t);
    uint32_t sizeDescale1 = 576 * sizeof(float);
    uint32_t sizeS2 = 32 * 576 * sizeof(float);

    // uint32_t sizeS3 = 2112 * sizeof(float);
    uint32_t sizeS3 = 32 * 576 * sizeof(float);

    uint32_t sizeSin1 = 32 * 64 * sizeof(half);
    uint32_t sizeS5 = 32* 576 * sizeof(float);
    uint32_t sizeCos1 = 32 * 64 * sizeof(half);
    uint32_t sizeKeycache1 = 192 * 128 * 512 * sizeof(half);
    uint32_t sizeKeycache2 = 192 * 128 * 64 * sizeof(float);
    uint32_t sizeQuantScale3 = 1 * sizeof(float);
    

    uint8_t *hostGamma3, *hostSlotMapping1, *hostDescale1, *hostS2, *hostS3, *hostSin1, *hostS5, *hostCos1, *hostQuantScale3, *hostTiling;
    uint8_t *deviceGamma3, *deviceSlotMapping1, *deviceDescale1, *deviceS2, *deviceS3, *deviceSin1, *deviceS5, *deviceCos1, *deviceQuantScale3,
        *deviceKeycache1, *deviceKeycache2, *deviceTiling;

    AllocMem(&hostGamma3, &deviceGamma3, sizeGamma3);
    ReadFile(dataPath + "/gamma3.bin", hostGamma3, sizeGamma3);
    ACL_CHECK(aclrtMemcpy(deviceGamma3, sizeGamma3, hostGamma3, sizeGamma3, ACL_MEMCPY_HOST_TO_DEVICE));

    AllocMem(&hostSlotMapping1, &deviceSlotMapping1, sizeSlotMapping1);
    ReadFile(dataPath + "/slotMapping.bin", hostSlotMapping1, sizeSlotMapping1);
    ACL_CHECK(aclrtMemcpy(deviceSlotMapping1, sizeSlotMapping1, hostSlotMapping1, sizeSlotMapping1, ACL_MEMCPY_HOST_TO_DEVICE));

    AllocMem(&hostDescale1, &deviceDescale1, sizeDescale1);
    ReadFile(dataPath + "/deScale11.bin", hostDescale1, sizeDescale1);
    ACL_CHECK(aclrtMemcpy(deviceDescale1, sizeDescale1, hostDescale1, sizeDescale1, ACL_MEMCPY_HOST_TO_DEVICE));

    AllocMem(&hostS2, &deviceS2, sizeS2);
    ReadFile(dataPath + "/s2.bin", hostS2, sizeS2);
    ACL_CHECK(aclrtMemcpy(deviceS2, sizeS2, hostS2, sizeS2, ACL_MEMCPY_HOST_TO_DEVICE));

    AllocMem(&hostS3, &deviceS3, sizeS3);//还没有golden
    ReadFile(dataPath + "/s2.bin", hostS3, sizeS3);
    ACL_CHECK(aclrtMemcpy(deviceS3, sizeS3, hostS3, sizeS3, ACL_MEMCPY_HOST_TO_DEVICE));

    AllocMem(&hostSin1, &deviceSin1, sizeSin1);
    ReadFile(dataPath + "/sin1.bin", hostSin1, sizeSin1);
    ACL_CHECK(aclrtMemcpy(deviceSin1, sizeSin1, hostSin1, sizeSin1, ACL_MEMCPY_HOST_TO_DEVICE));

    AllocMem(&hostS5, &deviceS5, sizeS5);
    ReadFile(dataPath + "/s5.bin", hostS5, sizeS5);
    ACL_CHECK(aclrtMemcpy(deviceS5, sizeS5, hostS5, sizeS5, ACL_MEMCPY_HOST_TO_DEVICE));

    AllocMem(&hostCos1, &deviceCos1, sizeCos1);
    ReadFile(dataPath + "/cos1.bin", hostCos1, sizeCos1);
    ACL_CHECK(aclrtMemcpy(deviceCos1, sizeCos1, hostCos1, sizeCos1, ACL_MEMCPY_HOST_TO_DEVICE));

    AllocMem(&hostQuantScale3, &deviceQuantScale3, sizeQuantScale3);
    ReadFile(dataPath + "/quantScale3.bin", hostS5, sizeS5);
    ACL_CHECK(aclrtMemcpy(deviceQuantScale3, sizeQuantScale3, hostQuantScale3, sizeQuantScale3, ACL_MEMCPY_HOST_TO_DEVICE));

    std::vector<int8_t> expectedKeycache1(sizeKeycache1);
    ReadFile(dataPath + "/keycache1.bin", expectedKeycache1.data(), sizeKeycache1);
    std::vector<int8_t> expectedKeycache2(sizeKeycache2);
    ReadFile(dataPath + "/keycache2.bin", expectedKeycache2.data(), sizeKeycache2);

    ACL_CHECK(aclrtMalloc((void **)&deviceKeycache1, sizeKeycache1, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&deviceKeycache2, sizeKeycache2, ACL_MEM_MALLOC_HUGE_FIRST));

    // get tiling
    uint32_t sizeTiling = sizeof(MlaPreprocessTilingData);
    AllocMem(&hostTiling, &deviceTiling, sizeTiling);

    MlaPreprocessTiling::MlaPreprocessInfo mpInfo;
    mpInfo.rmsNumCol2 = rmsNumCol2;
    MlaPreprocessTilingData mpTilingData;
    MlaPreprocessTiling::GetMpTilingParam(mpInfo, mpTilingData);
    hostTiling = reinterpret_cast<uint8_t *>(&mpTilingData);
    ACL_CHECK(aclrtMemcpy(deviceTiling, sizeTiling, hostTiling, sizeTiling, ACL_MEMCPY_HOST_TO_DEVICE));

    // Prepare FFTS address
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    if (dataType == "float16") {
        printf("MlaPreprocessFp16 start, aicoreNum = %d\n", aicoreNum);
        MlaPreprocessFp16<<<aicoreNum, nullptr, stream>>>(fftsAddr, deviceGamma3, deviceSlotMapping1, deviceDescale1, deviceS2, deviceS3, deviceSin1, deviceS5, deviceCos1, deviceKeycache1, deviceKeycache2, deviceQuantScale3, deviceTiling);
    } else {
        MlaPreprocessBf16<<<aicoreNum, nullptr, stream>>>(fftsAddr, deviceGamma3, deviceSlotMapping1, deviceDescale1, deviceS2, deviceS3, deviceSin1, deviceS5, deviceCos1, deviceKeycache1, deviceKeycache2, deviceQuantScale3, deviceTiling);
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));

    // std::vector<int8_t> hostKeycache1(sizeKeycache1);
    // ACL_CHECK(aclrtMemcpy(hostKeycache1.data(), sizeKeycache1, deviceKeycache1, sizeKeycache1, ACL_MEMCPY_DEVICE_TO_HOST));
    // std::vector<int8_t> hostKeycache2(sizeKeycache2);
    // ACL_CHECK(aclrtMemcpy(hostKeycache2.data(), sizeKeycache2, deviceKeycache2, sizeKeycache2, ACL_MEMCPY_DEVICE_TO_HOST));

    // std::vector<uint64_t> keycache1ErrorIndices = golden::CompareData(hostKeycache1, expectedKeycache1, 192 * 128 * 512);
    // std::vector<uint64_t> keycache2ErrorIndices = golden::CompareData(hostKeycache2, expectedKeycache2, 192 * 128 * 512);

    // if (keycache1ErrorIndices.empty()) {
    //     std::cout << "Compare out success." << std::endl;
    // } else {
    //     std::cerr << "Compare out failed. Error count: " << keycache1ErrorIndices.size() << std::endl;
    // }

    // if (keycache2ErrorIndices.empty()) {
    //     std::cout << "Compare out success." << std::endl;
    // } else {
    //     std::cerr << "Compare out failed. Error count: " << keycache2ErrorIndices.size() << std::endl;
    // }

    FreeMem(hostGamma3, deviceGamma3);
    FreeMem(hostSlotMapping1, deviceSlotMapping1);
    FreeMem(hostDescale1, deviceDescale1);
    FreeMem(hostS2, deviceS2);
    FreeMem(hostS3, deviceS3);
    FreeMem(hostSin1, deviceSin1);
    FreeMem(hostS5, deviceS5);
    FreeMem(hostCos1, deviceCos1);
    FreeMem(hostQuantScale3, deviceQuantScale3);

    aclrtFree(deviceKeycache1);
    aclrtFree(deviceKeycache2);

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