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
#include "catlass/status.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/kernel/w4a8_grouped_matmul_msd.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_broadcast_mul.hpp"
#include "catlass/epilogue/tile/tile_broadcast_add.hpp"
#include "catlass/epilogue/tile/tile_broadcast_one_blk.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"

#include "golden.hpp"
#include "helper.hpp"

using namespace Catlass;

using Options = GroupedGemmWithPerGroupOptions;

void Run(Options const &options)
{
    aclrtStream stream{nullptr};
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    auto aicoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    uint32_t problemCount = options.problemCount;
    uint32_t kGroupSize = options.kGroupSize;
    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    uint32_t k = options.problemShape.k();
    uint32_t quantGroupNum = k / kGroupSize;

    size_t lenA = static_cast<size_t>(m) * k;
    size_t lenB = static_cast<size_t>(k) * n * problemCount;
    size_t lenScale = static_cast<size_t>(n) * quantGroupNum * problemCount;
    size_t lenBias = static_cast<size_t>(n) * problemCount;
    size_t lenPerTokenScale = static_cast<size_t>(m);
    size_t lenD = static_cast<size_t>(m) * n;

    size_t sizeGroupList = problemCount * sizeof(int32_t);
    size_t sizeA = lenA * sizeof(int8_t);
    size_t sizeB = lenB * sizeof(int8_t) / 2;
    size_t sizeScale = lenScale * sizeof(uint64_t);
    size_t sizeBias = lenBias * sizeof(float);
    size_t sizePerTokenScale = lenPerTokenScale * sizeof(float);
    size_t sizeD = lenD * sizeof(bfloat16);
    size_t goldenSize = lenD * sizeof(float);

    using LayoutA = layout::RowMajor;
    using LayoutB = layout::zN;
    using LayoutScale = layout::RowMajor;
    using LayoutBias = layout::VectorLayout;
    using LayoutPerTokenScale = layout::VectorLayout;
    using LayoutD = layout::RowMajor;

    LayoutA layoutA{2 * m, k};
    LayoutB layoutB{k, n};
    LayoutScale layoutScale{quantGroupNum, n};
    LayoutBias layoutBias{n};
    LayoutPerTokenScale layoutPerTokenScale{m};
    LayoutD layoutD{m, n};

    void *hostGroupList = nullptr;
    ACL_CHECK(aclrtMallocHost(&hostGroupList, sizeGroupList));
    std::string inputGroupList_path = "../../examples/35_w4a8_grouped_matmul_msd/data/inputGroupList.dat";
    ReadFile(inputGroupList_path, hostGroupList, sizeGroupList);

    void *hostA = nullptr;
    ACL_CHECK(aclrtMallocHost(&hostA, sizeA));
    std::string inputA_path = "../../examples/35_w4a8_grouped_matmul_msd/data/inputA.dat";
    ReadFile(inputA_path, hostA, sizeA);

    void *hostB = nullptr;
    ACL_CHECK(aclrtMallocHost(&hostB, sizeB));
    std::string inputB_path = "../../examples/35_w4a8_grouped_matmul_msd/data/inputB.dat";
    ReadFile(inputB_path, hostB, sizeB);

    void *hostScale = nullptr;
    ACL_CHECK(aclrtMallocHost(&hostScale, sizeScale));
    std::string inputScale_path = "../../examples/35_w4a8_grouped_matmul_msd/data/inputScale.dat";
    ReadFile(inputScale_path, hostScale, sizeScale);

    void *hostBias = nullptr;
    ACL_CHECK(aclrtMallocHost(&hostBias, sizeBias));
    std::string inputBias_path = "../../examples/35_w4a8_grouped_matmul_msd/data/inputBias.dat";
    ReadFile(inputBias_path, hostBias, sizeBias);

    void *hostPerTokenScale = nullptr;
    ACL_CHECK(aclrtMallocHost(&hostPerTokenScale, sizePerTokenScale));
    std::string inputPerTokenScale_path = "../../examples/35_w4a8_grouped_matmul_msd/data/inputPerTokenScale.dat";
    ReadFile(inputPerTokenScale_path, hostPerTokenScale, sizePerTokenScale);

    std::vector<float> hExpected(goldenSize);
    std::string expected_path = "../../examples/35_w4a8_grouped_matmul_msd/data/expected.dat";
    ReadFile(expected_path, hExpected.data(), goldenSize);

    using ElementGroupList = int32_t;
    using ElementA = AscendC::int4b_t;
    using ElementB = AscendC::int4b_t;
    using ElementScale = uint64_t;
    using ElementBias = float;
    using ElementPerTokenScale = float;
    using ElementD = bfloat16;

    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);

    uint8_t *deviceGroupList, *deviceA, *deviceB, *deviceScale, *deviceBias, *devicePerTokenScale;
    uint8_t *deviceD;

    ACL_CHECK(aclrtMalloc((void **)&deviceGroupList, sizeGroupList, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&deviceA, sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&deviceB, sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&deviceScale, sizeScale, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&deviceBias, sizeBias, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&devicePerTokenScale, sizePerTokenScale, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&deviceD, sizeD, ACL_MEM_MALLOC_HUGE_FIRST));

    ACL_CHECK(aclrtMemcpy(deviceGroupList, sizeGroupList, hostGroupList, sizeGroupList,
        ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA, sizeA, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB, sizeB, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceScale, sizeScale, hostScale, sizeScale, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceBias, sizeBias, hostBias, sizeBias, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(devicePerTokenScale, sizePerTokenScale, hostPerTokenScale,
        sizePerTokenScale, ACL_MEMCPY_HOST_TO_DEVICE));

    using ArchTag = Arch::AtlasA2;
    constexpr uint32_t preloadStages = 1;
    constexpr uint32_t l1Stages = 2;
    constexpr uint32_t l0AStages = 2;
    constexpr uint32_t l0BStages = 2;
    constexpr uint32_t l0CStages = 1;
    constexpr uint32_t workspaceStages = 2;
    constexpr bool enableUnitFlag = false;
    constexpr bool enableShuffleK = true;
    using DispatchPolicy = Gemm::MmadAtlasA2W4A8GroupedMatmulMSD<
        preloadStages, l1Stages, l0AStages, l0BStages, l0CStages, enableUnitFlag, enableShuffleK>;

    using L1TileShape = GemmShape<128, 256, 512>;
    using L0TileShape = GemmShape<128, 256, 128>;

    using AType = Gemm::GemmType<ElementA, LayoutA>;
    using BType = Gemm::GemmType<ElementB, LayoutB>;
    using CType = Gemm::GemmType<half, layout::RowMajor>;

    using TileCopyMmad = Gemm::Tile::QuantTileCopyPerChannelPerGroup<ArchTag, AType, BType, CType>;
    using BlockMmad = Gemm::Block::BlockMmad<
        DispatchPolicy,
        L1TileShape, L0TileShape,
        AType, BType, CType,  void,
        TileCopyMmad
    >;

    using EpilogueDispatchPolicy = Epilogue::EpilogueAtlasA2W4A8GmmMSD;
    using BiasType = Gemm::GemmType<ElementBias, LayoutBias>;
    using PerTokenScaleType = Gemm::GemmType<ElementPerTokenScale, LayoutPerTokenScale>;
    using DType = Gemm::GemmType<ElementD, LayoutD>;

    using RowBroadcastAddType = Gemm::GemmType<float, layout::RowMajor>;
    using BroadcastOneBlkType = Gemm::GemmType<float, layout::RowMajor>;
    using OneBlkColumnBroadcastMulType = Gemm::GemmType<float, layout::RowMajor>;

    using EpilogueTileShape = MatrixShape<64, 256>;
    using TileRowBroadcastAdd = Epilogue::Tile::TileRowBroadcastAdd<ArchTag, RowBroadcastAddType, EpilogueTileShape>;
    using TileBroadcastOneBlk =
        Epilogue::Tile::TileBroadcastOneBlk<ArchTag, BroadcastOneBlkType, EpilogueTileShape::ROW>;
    using TileOneBlkColumnBroadcastMul =
        Epilogue::Tile::TileOneBlkColumnBroadcastMul<ArchTag, OneBlkColumnBroadcastMulType, EpilogueTileShape>;
    using TileCopy = Epilogue::Tile::TileCopyW4A8GmmMSD<ArchTag, CType, BiasType, PerTokenScaleType, DType>;
    using TileScheduler = Epilogue::Tile::EpilogueHorizontalTileSwizzle;

    using BlockEpilogue = Epilogue::Block::BlockEpilogue<
        EpilogueDispatchPolicy, CType, BiasType, PerTokenScaleType, DType, TileRowBroadcastAdd, TileBroadcastOneBlk,
        TileOneBlkColumnBroadcastMul, TileCopy, TileScheduler>;

    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

    // kernel level
    using MatmulKernel = Gemm::Kernel::W4A8GroupedMatmulMSD<
        BlockMmad, BlockEpilogue, BlockScheduler, workspaceStages, ElementGroupList>;
    // call a kernel
    using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;

    MatmulAdapter matmulOp;
    typename MatmulKernel::Arguments arguments{
        options.problemShape,
        problemCount,
        kGroupSize,
        quantGroupNum,
        aicoreNum,
        deviceGroupList, 
        deviceA, layoutA,
        deviceB, layoutB,
        deviceScale, layoutScale,
        deviceBias, layoutBias,
        devicePerTokenScale, layoutPerTokenScale,
        deviceD, layoutD
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
   
    ACL_CHECK(aclrtSynchronizeStream(stream));

    ACL_CHECK(aclrtFree(deviceGroupList));
    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceScale));
    ACL_CHECK(aclrtFree(deviceBias));
    ACL_CHECK(aclrtFree(devicePerTokenScale));

    std::vector<bfloat16> hostD(sizeD);
    ACL_CHECK(aclrtMemcpy(hostD.data(), sizeD, deviceD, sizeD, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtFree(deviceD));

    std::vector<uint64_t> errorIndices = golden::CompareData(hostD, hExpected, k, m * n);
    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    ACL_CHECK(aclrtFreeHost(hostGroupList));
    ACL_CHECK(aclrtFreeHost(hostA));
    ACL_CHECK(aclrtFreeHost(hostB));
    ACL_CHECK(aclrtFreeHost(hostScale));
    ACL_CHECK(aclrtFreeHost(hostBias));
    ACL_CHECK(aclrtFreeHost(hostPerTokenScale));

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