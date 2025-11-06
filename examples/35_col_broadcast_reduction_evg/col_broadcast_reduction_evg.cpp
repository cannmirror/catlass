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

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_elemwise_add.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/matmul_epilogue.hpp"
#include "catlass/gemm/kernel/matmul_visitor.hpp"
#include "catlass/epilogue/fusion/fusion.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"

#include "golden.hpp"
#include "helper.hpp"

using namespace Catlass;

using Options = GemmOptions;

static void Run(const Options &options) {
    aclrtStream stream{nullptr};

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    uint32_t k = options.problemShape.k();

    // Compute the length of each matrix and the size of each buffer
    size_t lenA = static_cast<size_t>(m) * k;
    size_t lenB = static_cast<size_t>(k) * n;
    size_t lenD = static_cast<size_t>(m) * n;
    size_t lenX = lenD;

    size_t sizeA = lenA * sizeof(fp16_t);
    size_t sizeB = lenB * sizeof(fp16_t);
    size_t sizeD = lenD * sizeof(fp16_t);

    // Define the layout of each matrix
    using LayoutA = layout::RowMajor;
    using LayoutB = layout::RowMajor;
    using LayoutC = layout::RowMajor;
    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n};
    LayoutC layoutD{m, n};

    // Prepare input data A, B, and X
    std::vector<fp16_t> hostA(lenA);
    std::vector<fp16_t> hostB(lenB);
    std::vector<fp16_t> hostX(lenX);
    std::vector<fp16_t> hostCol(m);     // col broadcast input (1 x m)
    std::vector<float> hostColOut(m);   // col reduction output (1 x m), float for accumulation
    std::srand(std::time(nullptr));
    // golden::FillRandomData<fp16_t>(hostA, -5.0f, 5.0f);
    // golden::FillRandomData<fp16_t>(hostB, -5.0f, 5.0f);
    // golden::FillRandomData<fp16_t>(hostX, -5.0f, 5.0f);
    // golden::FillRandomData<fp16_t>(hostRow, -2.0f, 2.0f);
    // golden::FillRandomData<fp16_t>(hostA, 1.0f, 1.0f);
    // golden::FillRandomData<fp16_t>(hostB, 1.0f, 1.0f);
    // golden::FillRandomData<fp16_t>(hostX, 1.0f, 1.0f);
    // golden::FillRandomData<fp16_t>(hostRow, 1.0f, 1.0f);

    auto FillRandomIntDataToFp16 = [](std::vector<fp16_t>& data, int32_t low, int32_t high) {
        for (uint64_t i = 0; i < data.size(); ++i) {
            data[i] = static_cast<fp16_t>(low + rand() % (high - low + 1));
        }
    };

    FillRandomIntDataToFp16(hostA, -5, 5);
    FillRandomIntDataToFp16(hostB, -5, 5);
    FillRandomIntDataToFp16(hostX, -5, 5);
    FillRandomIntDataToFp16(hostCol, -5, 5);

    // Allocate device memory and copy data from host to device
    uint8_t *deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate separate memory for X, Row and D
    uint8_t *deviceX{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceX), sizeD, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceX, sizeD, hostX.data(), sizeD, ACL_MEMCPY_HOST_TO_DEVICE));

    size_t sizeCol = static_cast<size_t>(m) * sizeof(fp16_t);
    size_t sizeColOut = static_cast<size_t>(m) * sizeof(float);
    uint8_t *deviceCol{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceCol), sizeCol, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceCol, sizeCol, hostCol.data(), sizeCol, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceColOut{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceColOut), sizeColOut, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemset(deviceColOut, sizeColOut, 0, sizeColOut));

    uint8_t *deviceD{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceD), sizeD, ACL_MEM_MALLOC_HUGE_FIRST));

    // Prepare FFTS address
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    // Get the number of cube cores of the current hardware
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    // Define ArchTag
    using ArchTag = Arch::AtlasA2;

    // Block level, define BlockMmad
    constexpr bool enableUnitFlag = true;
    using MmadDispatchPolicy = Gemm::MmadAtlasA2Pingpong<enableUnitFlag>;
    using L1TileShape = GemmShape<128, 256, 256>;
    using L0TileShape = GemmShape<128, 256, 64>;
    using AType = Gemm::GemmType<half, LayoutA>;
    using BType = Gemm::GemmType<half, LayoutB>;
    using CType = Gemm::GemmType<half, LayoutC>;
    using BlockMmad = Gemm::Block::BlockMmad<MmadDispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;

    // 定义 EVG: D = C + ColBroadcast(col); 同时对 (C+col) 做按列规约并加到 deviceColOut
    constexpr uint32_t computeLength = 4096;
    
    using EVG = Epilogue::Fusion::TreeVisitor<
        Epilogue::Fusion::VisitorColReduce<Epilogue::Fusion::Plus, float, layout::RowMajor>,
        Epilogue::Fusion::TreeVisitor<
            Epilogue::Fusion::VisitorCast<float, half>,
            Epilogue::Fusion::TreeVisitor<
                Epilogue::Fusion::VisitorAuxStore<half, LayoutC>,
                Epilogue::Fusion::TreeVisitor<
                    Epilogue::Fusion::VisitorCompute<Epilogue::Fusion::Plus, half>,
                    Epilogue::Fusion::VisitorAccLoad<half>,
                    Epilogue::Fusion::VisitorColBroadcast<half, layout::RowMajor>
                >
            >
        >
    >;

    // Block level, define BlockEpilogue with EVG
    using BlockEpilogue = Epilogue::Block::BlockEpilogue<
        Epilogue::EpilogueAtlasA2Visitor,
        CType,
        tla::Int<computeLength>,
        EVG
    >;

    // 准备 EVG Arguments
    typename EVG::Arguments evg_args{
        {
            {
                {
                    {},
                    {deviceCol, layout::RowMajor{m, 1}},
                    {}
                },
                {deviceD, layoutD}
            },
            {}
        },
        {deviceColOut, layout::RowMajor{m, 1}, 0.0f}
    };

    std::vector<fp16_t> hostD(lenD);
    // Define BlockScheduler
    // Swizzle offset is 3 and direction is 0.
    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
    // Kernel level
    using MatmulKernel = Gemm::Kernel::MatmulVisitor<BlockMmad, BlockEpilogue, BlockScheduler>;
    // Prepare params
    typename MatmulKernel::Arguments arguments{options.problemShape, deviceA, deviceB, evg_args};
    using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
    MatmulAdapter matmulOp;
    size_t sizeWorkspace = matmulOp.GetWorkspaceSize(arguments);
    uint8_t *deviceWorkspace{nullptr};
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST)
        );
    }
    matmulOp.Initialize(arguments, deviceWorkspace);
    matmulOp(stream, aicCoreNum, fftsAddr);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtFree(deviceWorkspace));
    }


    // Copy the result from device to host
    ACL_CHECK(aclrtMemcpy(hostD.data(), sizeD, deviceD, sizeD, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(hostColOut.data(), sizeColOut, deviceColOut, sizeColOut, ACL_MEMCPY_DEVICE_TO_HOST));

    // Compute the golden result
    std::vector<float> hostGolden(lenD);
    // Golden for D: C + col (col-broadcast add)
    std::vector<fp16_t> hostColBroadcast(lenD);
    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < n; ++j) {
            hostColBroadcast[static_cast<size_t>(i) * n + j] = hostCol[i];
        }
    }
    golden::ComputeMatmulElemWiseAdd(options.problemShape, hostA, layoutA, hostB, layoutB, hostColBroadcast, hostGolden, layoutD);


    // Compare the matrix result
    std::vector<uint64_t> errorIndices = golden::CompareData(hostD, hostGolden, k);
    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
        for (size_t i = 0; i < min(10, errorIndices.size()); ++i) {
            std::cerr << "  Error[" << i << "] = " << errorIndices[i] << " (D[" << errorIndices[i] << "] = " << static_cast<float>(hostD[errorIndices[i]]) << ", Golden[" << errorIndices[i] << "] = " << static_cast<float>(hostGolden[errorIndices[i]]) << ")" << std::endl;
        }
    }

    // Golden for col reduction: sum over cols of (C + col)
    std::vector<float> goldenColOut(m, 0.0f);
    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < n; ++j) {
            goldenColOut[i] += static_cast<float>(hostGolden[i * n + j]);
        }
    }
    // Compare col reduce output
    std::vector<uint64_t> errCol = golden::CompareData(hostColOut, goldenColOut, k);
    if (errCol.empty()) {
        std::cout << "ColReduce compare success." << std::endl;
    } else {
        std::cerr << "ColReduce compare failed. Error count: " << errCol.size() << std::endl;
        for (size_t i = 0; i < min(10, errCol.size()); ++i) {
            std::cerr << "  Error[" << i << "] = " << errCol[i] << " (ColOut[" << errCol[i] << "] = " << hostColOut[errCol[i]] << ", GoldenCol[" << errCol[i] << "] = " << goldenColOut[errCol[i]] << ")" << std::endl;
        }
    }

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceX));
    ACL_CHECK(aclrtFree(deviceCol));
    ACL_CHECK(aclrtFree(deviceColOut));
    ACL_CHECK(aclrtFree(deviceD));

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
}

int main(int argc, const char **argv) {
    Options options;
    if (options.Parse(argc, argv) != 0) {
        return -1;
    }
    Run(options);
    return 0;
}
