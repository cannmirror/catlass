/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#ifndef CATLASS_CUSTOM_BLOCK_BLOCK_MMAD_CUSTOM_HPP
#define CATLASS_CUSTOM_BLOCK_BLOCK_MMAD_CUSTOM_HPP

#include "catlass/catlass.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"

namespace Catlass::CustomGemm::Block {
struct BlockMmadCustom {
    
}
    template <
    bool ENABLE_UNIT_FLAG_,
    class L1TileShape_,
    class L0TileShape_,
    class AType_,
    class BType_,
    class CType_,
    class BiasType_,
    class TileCopy_,
    class TileMmad_
>
struct BlockMmad <
    MmadAtlasA2Pingpong<ENABLE_UNIT_FLAG_>,
    GemmShape<32, 32, 32>,
    GemmShape<32, 32, 32>,
    GemmType<half, layout::RowMajor>,
    GemmType<half, layout::RowMajor>, 
    GemmType<half, layout::RowMajor>,
    BiasType_,
    TileCopy_,
    TileMmad_
> 
} // namespace Custom::Block