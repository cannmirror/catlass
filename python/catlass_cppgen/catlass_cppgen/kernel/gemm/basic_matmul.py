from catlass_cppgen.kernel.gemm.gemm_base import GemmKernelBase


class BasicMatmulKernel(GemmKernelBase):
    _TEMPLATE = """\
#include <catlass/gemm/kernel/basic_matmul.hpp>
#include <catlass/arch/arch.hpp>
#include <catlass/catlass.hpp>
#include <catlass/gemm/block/block_mmad.hpp>
#include <catlass/gemm/block/block_swizzle.hpp>
#include <catlass/gemm/device/device_gemm.hpp>
#include <catlass/gemm/dispatch_policy.hpp>
#include <catlass/gemm/gemm_type.hpp>
#include <catlass/layout/layout.hpp>

extern "C" __global__ __aicore__ void basic_matmul_kernel(GemmCoord problemShape, GM_ADDR ptrA, GM_ADDR ptrB, GM_ADDR ptrC) {{
    using ArchTag = {arch_tag};
    using DispatchPolicy = Gemm::MmadAtlasA2Pingpong<true>;
    using L1TileShape = {l1_tile_shape};
    using L0TileShape = {l0_tile_shape};
    using LayoutA = {layout_A};
    using LayoutB = {layout_B};
    using LayoutC = layout::RowMajor;
    using AType = Gemm::GemmType<{element_A}, LayoutA>;
    using BType = Gemm::GemmType<{element_B}, LayoutB>;
    using CType = Gemm::GemmType<{element_C}, LayoutC>;
    using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
    using BlockEpilogue = void;
    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
    using MatmulKernel = Gemm::Kernel::BasicMatmul<BlockMmad, BlockEpilogue, BlockScheduler>;
    LayoutA layoutA{{problemShape.m(), problemShape.k()}};
    LayoutB layoutB{{problemShape.k(), problemShape.n()}};
    LayoutC layoutC{{problemShape.m(), problemShape.n()}};
    typename MatmulKernel::Params params{{problemShape, gmA, layoutA, gmB, layoutB, gmC, layoutC}};
    MatmulKernel kernel;
    kernel(params);
}}
    """
