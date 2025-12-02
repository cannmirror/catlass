# 矩阵乘模板总结

当前库上`examples`内包含多种矩阵乘的`样例模板`，其来源是不同的matmul`理论模板`与工程实践中发现的`工程优化`点的组合。在充分理解了各个`理论模板`和`工程优化`后，开发者可以基于问题场景选择适合的`样例模板`、甚至进一步自行组合出库上没有的新的`样例模板`，来达到矩阵乘的性能极致优化。

## 样例模板清单
<details>
<summary><strong>00_basic_matmul</strong></summary>

- 理论模板：`Common模板`
- 工程优化：无
- 关键交付件
    - host：[00_basic_matmul](../../../examples/00_basic_matmul/basic_matmul.cpp)
    - kernel：[basic_matmul.hpp](../../../include/catlass/gemm/kernel/basic_matmul.hpp)
    - blockMmad：[block_mmad_pingpong.hpp](../../../include/catlass/gemm/block/block_mmad_pingpong.hpp)
- dispatchPolicy：`MmadAtlasA2Pingpong`
</details>

<details>
<summary><strong>04_padding_matmul</strong></summary>

- 理论模板：`Common模板`
- 工程优化：`读取带宽优化（padding）- PaddingMatrixND`
- 关键交付件
    - host：[04_padding_matmul](../../../examples/04_padding_matmul/padding_matmul.cpp)
    - kernel：[padding_matmul.hpp](../../../include/catlass/gemm/kernel/padding_matmul.hpp)
    - blockMmad：[block_mmad_pingpong.hpp](../../../include/catlass/gemm/block/block_mmad_pingpong.hpp)
- dispatchPolicy：`MmadAtlasA2Pingpong`
</details>

<details>
<summary><strong>06_optimized_matmul</strong></summary>

- 理论模板：`Common模板`
- 工程优化：
    - `流水优化（Preload）`
    - `读取带宽优化（Padding）- PaddingMatrixNZ`
    - `读取带宽优化（ShuffleK）`
    - `读取带宽优化（小M下指令替换）`
- 关键交付件
    - host：[06_optimized_matmul](../../../examples/06_optimized_matmul/optimized_matmul.cpp)
    - kernel：[optimized_matmul.hpp](../../../include/catlass/gemm/kernel/optimized_matmul.hpp)
    - Padding前处理组件：[padding_matmul.hpp](../../../include/catlass/gemm/kernel/padding_matmul.hpp)
    - blockMmad：[block_mmad_preload.hpp](../../../include/catlass/gemm/block/block_mmad_preload.hpp)
- dispatchPolicy：`MmadAtlasA2Preload`
</details>

<details>
<summary><strong>09_splitk_matmul</strong></summary>

- 理论模板：`多核切K模板 MultiCoreSplitK`
- 工程优化：无
- 关键交付件
    - host：[09_splitk_matmul](../../../examples/09_splitk_matmul/optimized_matmul.cpp)
    - kernel：[splitk_matmul.hpp](../../../include/catlass/gemm/kernel/splitk_matmul.hpp)
    - blockMmad：[block_mmad_pingpong.hpp](../../../include/catlass/gemm/block/block_mmad_pingpong.hpp)
- dispatchPolicy：`MmadAtlasA2Pingpong`
</details>

<details>
<summary><strong>21_basic_matmul_preload_zN</strong></summary>

（此样例主要承载NZ排布输入的适配方法）
- 理论模板：`Common模板`
- 工程优化：
    - `流水优化（Preload）`
    - `读取带宽优化（ShuffleK）`
- 关键交付件
    - host：[21_basic_matmul_preload_zN](../../../examples/09_splitk_matmul/basic_matmul_preload_zN.cpp)
    - kernel：[basic_matmul_preload.hpp](../../../include/catlass/gemm/kernel/basic_matmul_preload.hpp)
    - blockMmad：[block_mmad_preload.hpp](../../../include/catlass/gemm/block/block_mmad_preload.hpp)
- dispatchPolicy：`MmadAtlasA2Preload`
</details>

<details>
<summary><strong>22_padding_splitk_matmul</strong></summary>

- 理论模板：`多核切K模板 MultiCoreSplitK`
- 工程优化：`读取带宽优化（padding）- PaddingMatrixND`
- 关键交付件
    - host：[22_padding_splitk_matmul](../../../examples/22_padding_splitk_matmul/padding_splitk_matmul.cpp)
    - kernel：[padding_splitk_matmul.hpp](../../../include/catlass/gemm/kernel/padding_splitk_matmul.hpp)
    - Padding前处理组件：[padding_matmul.hpp](../../../include/catlass/gemm/kernel/padding_matmul.hpp)
    - SplitkReduceAdd后处理组件：[splitk_matmul.hpp](../../../include/catlass/gemm/kernel/splitk_matmul.hpp)
    - blockMmad：[block_mmad_pingpong.hpp](../../../include/catlass/gemm/block/block_mmad_pingpong.hpp)
- dispatchPolicy：`MmadAtlasA2Pingpong`
</details>

<details>
<summary><strong>25_matmul_full_loadA</strong></summary>

（此样例及相关组件仅适配了A矩阵全载实现，需要实现B矩阵全载可参考关键交付件自行开发）
- 理论模板：`Common模板`
- 工程优化：
    - `读取带宽优化（L1常驻）`
- 关键交付件
    - host：[25_matmul_full_loadA](../../../examples/09_splitk_matmul/25_matmul_full_loadA.cpp)
    - kernel：[matmul_full_loadA.hpp](../../../include/catlass/gemm/kernel/matmul_full_loadA.hpp)
    - blockMmad：[block_mmad_pingpong_full_loadA.hpp](../../../include/catlass/gemm/block/block_mmad_pingpong_full_loadA.hpp)
- dispatchPolicy：`MmadAtlasA2FullLoadA`
- BlockScheduler：`GemmIdentityBlockSwizzleL1FullLoad`
</details>

<details>
<summary><strong>31_small_matmul</strong></summary>

- 理论模板：`Common模板`
- 工程优化：
    - `Scalar开销消减`
- 关键交付件
    - host：[31_small_matmul](../../../examples/31_small_matmul/small_matmul.cpp)
    - kernel：[small_matmul.hpp](../../../include/catlass/gemm/kernel/small_matmul.hpp)
    - blockMmad：[block_mmad_small.hpp](../../../include/catlass/gemm/block/block_mmad_small.hpp)
- dispatchPolicy：`MmadAtlasA2Small`
- BlockScheduler：kernel内实际不使用
</details>

<details>
<summary><strong>34_single_core_splitk_matmul</strong></summary>

- 理论模板：`单核切K模板 SingleCoreSplitK`
- 工程优化：
    - `读取带宽优化（Padding）- PaddingMatrixNZ`
    - `写出带宽优化`
- 关键交付件
    - host：[34_single_core_splitk_matmul](../../../examples/34_single_core_splitk_matmul/single_core_splitk.cpp)
    - kernel：[single_core_slicek_matmul.hpp](../../../include/catlass/gemm/kernel/single_core_slicek_matmul.hpp)
    - blockMmad：[block_mmad_single_core_splitk.hpp](../../../include/catlass/gemm/block/block_mmad_single_core_splitk.hpp)
- dispatchPolicy：`MmadAtlasA2SingleCoreSplitk`
- BlockScheduler：`SingleCoreSplitkGemmIdentityBlockSwizzle`
</details>

## 理论模板清单

<details>
<summary><strong>Common模板</strong></summary>

</details>

<details>
<summary><strong>多核切K模板 MultiCoreSplitK</strong></summary>

</details>

<details>
<summary><strong>单核切K模板 SingleCoreSplitK</strong></summary>

</details>

## 工程优化清单

<details>
<summary><strong>流水优化（Preload）</strong></summary>

### 现象分析
### 优化方案

</details>

<details>
<summary><strong>读取带宽优化（Padding）</strong></summary>

### 现象分析
### 优化方案
#### PaddingMatrixND
#### PaddingMatrixBlockND
#### PaddingMatrixNZ

</details>

<details>
<summary><strong>读取带宽优化（ShuffleK）</strong></summary>

### 现象分析
### 优化方案

</details>

<details>
<summary><strong>读取带宽优化（小M下指令替换）</strong></summary>

### 现象分析
### 优化方案

</details>

<details>
<summary><strong>读取带宽优化（L1常驻）</strong></summary>

### 现象分析
### 优化方案

</details>

<details>
<summary><strong>Scalar开销消减</strong></summary>

### 现象分析
### 优化方案

</details>

<details>
<summary><strong>写出带宽优化</strong></summary>

### 现象分析
### 优化方案

</details>