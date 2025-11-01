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
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <cstring>
#include <iostream>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <string>

#include <tiling/platform/platform_ascendc.h>
#include <acl/acl.h>
#include <runtime/rt_ffts.h>

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/kernel/w4a8_grouped_matmul_slice_m.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/device/device_gemm.hpp"

#include "golden.hpp"
#include "helper.hpp"
using namespace Catlass;

using Options = GroupedGemmOptions;
static void Run(Options const &options)
{
    aclrtStream stream{nullptr};
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    auto aicoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    uint32_t problemCount = options.problemCount;
    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    uint32_t k = options.problemShape.k();

    using LayoutA = layout::RowMajor;
    using LayoutPrologueB = layout::RowMajor;
    using LayoutB = layout::RowMajor;
    using LayoutC = layout::RowMajor;

    LayoutA layoutA{m, k};
    LayoutPrologueB layoutPrologueB{k, n, (n + 1) / 2 * 2};
    LayoutC layoutC{m, n};

    uint64_t sizeGroupList = problemCount * sizeof(int64_t);
    uint64_t sizeScales = problemCount * sizeof(float);
    uint64_t sizeA = layoutA.Capacity() * sizeof(int8_t);
    uint64_t sizeB = layoutPrologueB.Capacity() / 2 * sizeof(int8_t) * problemCount;
    uint64_t sizeC = layoutC.Capacity() * sizeof(__fp16);
    uint64_t goldenSize = layoutC.Capacity() * sizeof(float);

    void *hostGroupList{nullptr};
    ACL_CHECK(aclrtMallocHost(&hostGroupList, sizeGroupList));
    std::string inputGroupList_path = "../../examples/33_w4a8_grouped_matmul_slice_m/data/inputGroupList.dat";
    ReadFile(inputGroupList_path, hostGroupList, sizeGroupList);

    void *hostScales{nullptr};
    ACL_CHECK(aclrtMallocHost(&hostScales, sizeScales));
    std::string inputScales_path = "../../examples/33_w4a8_grouped_matmul_slice_m/data/inputScales.dat";
    ReadFile(inputScales_path, hostScales, sizeScales);

    void *hostA = nullptr;
    ACL_CHECK(aclrtMallocHost(&hostA, sizeA));
    std::string inputA_path = "../../examples/33_w4a8_grouped_matmul_slice_m/data/inputA.dat";
    ReadFile(inputA_path, hostA, sizeA);

    void *hostB = nullptr;
    ACL_CHECK(aclrtMallocHost(&hostB, sizeB));
    std::string inputB_path = "../../examples/33_w4a8_grouped_matmul_slice_m/data/inputB.dat";
    ReadFile(inputB_path, hostB, sizeB);

    std::vector<float> hExpected(goldenSize);
    std::string expected_path = "../../examples/33_w4a8_grouped_matmul_slice_m/data/expected.dat";
    ReadFile(expected_path, hExpected.data(), goldenSize);

    using ElementA = int8_t;
    using ElementPrologueB = AscendC::int4b_t;
    using ElementB = int8_t;
    using ElementC = half;

    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);

    uint8_t *deviceGroupList, *deviceScales, *deviceA, *deviceB, *deviceC;

    ACL_CHECK(aclrtMalloc((void **)&deviceGroupList, sizeGroupList, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&deviceScales, sizeScales, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&deviceA, sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&deviceB, sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&deviceC, sizeC, ACL_MEM_MALLOC_HUGE_FIRST));

    ACL_CHECK(aclrtMemcpy(deviceGroupList, sizeGroupList, hostGroupList,
        sizeGroupList, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceScales, sizeScales, hostScales, sizeScales, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA, sizeA, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB, sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    using ArchTag = Arch::AtlasA2;

    constexpr bool enableUnitFlag = false;
    using DispatchPolicy = Gemm::MmadAtlasA2PingPongWithPrologue<enableUnitFlag>;

    // if LayoutA and LayoutB is both ColumnMajor
    // L1TileShape using MatmulShape<256, 128, 256> can achieve better performance.
    using L1TileShape = std::conditional_t<std::is_same_v<LayoutA, layout::ColumnMajor> &&
        std::is_same_v<LayoutB, layout::ColumnMajor>, GemmShape<256, 128, 512>, GemmShape<128, 256, 512>>;
    using L0TileShape = std::conditional_t<std::is_same_v<LayoutA, layout::ColumnMajor> &&
        std::is_same_v<LayoutB, layout::ColumnMajor>, GemmShape<256, 128, 128>, GemmShape<128, 256, 128>>;
    using PrologueSrcType = Gemm::GemmType<ElementPrologueB, LayoutPrologueB>;
    using PrologueDstType = Gemm::GemmType<ElementB, LayoutB>;

    using AType = Gemm::GemmType<ElementA, LayoutA>;
    using BType = PrologueDstType;
    using CType = Gemm::GemmType<ElementC, LayoutC>;

    using PrologueA = void;
    constexpr uint32_t computeLen = 24 * 1024;  // todo
    using PrologueB = Gemm::Tile::TileCastInt4ToInt8<ArchTag, PrologueSrcType, PrologueDstType, computeLen>;

    using TileCopy = Gemm::Tile::TileCopyWithPrologueDeqPerTensor<ArchTag, AType, BType, CType, PrologueA, PrologueB>;
    using BlockMmad = Gemm::Block::BlockMmad<
        DispatchPolicy,
        L1TileShape, L0TileShape,
        AType, BType, CType,  void,
        TileCopy
    >;

    if (options.problemShape.m() > options.problemShape.n()) {  // todo
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
        using BlockEpilogue = void;
        // kernel level
        using MatmulKernel = Gemm::Kernel::W4A8GroupedMatmulSliceM<BlockMmad, BlockEpilogue, BlockScheduler, int64_t>;
        // call a kernel
        using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;

        MatmulAdapter matmulOp;
        typename MatmulKernel::Arguments arguments{
            options.problemShape, problemCount, deviceGroupList,
            deviceA, layoutA,
            deviceB, layoutPrologueB,
            deviceC, layoutC,
            deviceScales,
            aicoreNum
        };
        matmulOp.CanImplement(arguments);

        size_t sizeWorkspace = matmulOp.GetWorkspaceSize(arguments);
        uint8_t *deviceWorkspace = nullptr;
        if (sizeWorkspace > 0) {
            ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
        }

        matmulOp.Initialize(arguments, deviceWorkspace);
        matmulOp(stream, aicoreNum, fftsAddr);
        if (sizeWorkspace > 0) {
            ACL_CHECK(aclrtFree(deviceWorkspace));
        }
    } else {
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
        using BlockEpilogue = void;
        // kernel level
        using MatmulKernel = Gemm::Kernel::W4A8GroupedMatmulSliceM<BlockMmad, BlockEpilogue, BlockScheduler, int64_t>;
        // call a kernel
        using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
        MatmulAdapter matmulOp;
        typename MatmulKernel::Arguments arguments{
            options.problemShape, problemCount, deviceGroupList,
            deviceA, layoutA,
            deviceB, layoutPrologueB,
            deviceC, layoutC,
            deviceScales,
            aicoreNum
        };
        matmulOp.CanImplement(arguments);

        size_t sizeWorkspace = matmulOp.GetWorkspaceSize(arguments);
        uint8_t *deviceWorkspace = nullptr;
        if (sizeWorkspace > 0) {
            ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
        }

        matmulOp.Initialize(arguments, deviceWorkspace);
        matmulOp(stream, aicoreNum, fftsAddr);
        if (sizeWorkspace > 0) {
            ACL_CHECK(aclrtFree(deviceWorkspace));
        }
    }

    ACL_CHECK(aclrtSynchronizeStream(stream));

    ACL_CHECK(aclrtFree(deviceGroupList));
    ACL_CHECK(aclrtFree(deviceScales));
    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));

    std::vector<fp16_t> hostC(sizeC);
    ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtFree(deviceC));

    std::vector<uint64_t> errorIndices = golden::CompareData(hostC, hExpected, k);
    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    ACL_CHECK(aclrtFreeHost(hostGroupList));
    ACL_CHECK(aclrtFreeHost(hostScales));
    ACL_CHECK(aclrtFreeHost(hostA));
    ACL_CHECK(aclrtFreeHost(hostB));

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