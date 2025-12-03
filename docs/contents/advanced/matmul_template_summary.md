# 矩阵乘模板总结

当前库上`examples`内包含多种矩阵乘的`样例模板`，其来源是不同的matmul`理论模板`与工程实践中发现的`工程优化`点的组合。在充分理解了各个`理论模板`和`工程优化`后，开发者可以基于问题场景选择适合的`样例模板`、甚至进一步自行组合出库上没有的新的`样例模板`，来达到矩阵乘的性能极致优化。

## 样例模板清单
<details>
<summary><strong><font size="4">00_basic_matmul</font></strong></summary>

- 理论模板：`Common模板`
- 工程优化：无
- 关键交付件
    - host：[00_basic_matmul](../../../examples/00_basic_matmul/basic_matmul.cpp)
    - kernel：[basic_matmul.hpp](../../../include/catlass/gemm/kernel/basic_matmul.hpp)
    - blockMmad：[block_mmad_pingpong.hpp](../../../include/catlass/gemm/block/block_mmad_pingpong.hpp)
- dispatchPolicy：`MmadAtlasA2Pingpong`
</details>

<details>
<summary><strong><font size="4">04_padding_matmul</font></strong></summary>

- 理论模板：`Common模板`
- 工程优化：`读取带宽优化（padding）- PaddingMatrixND`
- 关键交付件
    - host：[04_padding_matmul](../../../examples/04_padding_matmul/padding_matmul.cpp)
    - kernel：[padding_matmul.hpp](../../../include/catlass/gemm/kernel/padding_matmul.hpp)
    - blockMmad：[block_mmad_pingpong.hpp](../../../include/catlass/gemm/block/block_mmad_pingpong.hpp)
- dispatchPolicy：`MmadAtlasA2Pingpong`
</details>

<details>
<summary><strong><font size="4">06_optimized_matmul</font></strong></summary>

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
<summary><strong><font size="4">09_splitk_matmul</font></strong></summary>

- 理论模板：`多核切K模板 MultiCoreSplitK`
- 工程优化：无
- 关键交付件
    - host：[09_splitk_matmul](../../../examples/09_splitk_matmul/optimized_matmul.cpp)
    - kernel：[splitk_matmul.hpp](../../../include/catlass/gemm/kernel/splitk_matmul.hpp)
    - blockMmad：[block_mmad_pingpong.hpp](../../../include/catlass/gemm/block/block_mmad_pingpong.hpp)
- dispatchPolicy：`MmadAtlasA2Pingpong`
</details>

<details>
<summary><strong><font size="4">21_basic_matmul_preload_zN</font></strong></summary>

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
<summary><strong><font size="4">22_padding_splitk_matmul</font></strong></summary>

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
<summary><strong><font size="4">25_matmul_full_loadA</font></strong></summary>

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
<summary><strong><font size="4">31_small_matmul</font></strong></summary>

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
<summary><strong><font size="4">34_single_core_splitk_matmul</font></strong></summary>

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
<summary><strong><font size="4">Common模板</font></strong></summary>

### Tiling建模

<img src="https://raw.gitcode.com/user-images/assets/7801479/b1cb21ac-af83-4736-8582-4ed7392d766b/1common.png" width="80%">

如图展示一个常规的fp16的矩阵运算（L0C上按照fp32累加），定义相关参数：
- 问题shape：$M$，$N$，$K$
- 搬入L1Cache时的TileShape：$m_1$，$n_1$，$k_1$
- 搬入L0A/LOB、搬出L0C时的TileShape：$m_0$，$n_0$，$k_0$

采用 $M$、$N$ 方向分核，产生$\frac{MN}{m_1n_1}$个基本任务块、分配给AIC核完成搬运和计算，每个基本任务块需要搬运$m_1K+Kn_1$的数据、计算得到$m_1n_1$的结果并搬出。由此产生约束：
- $m_1k_1*L1Stage_A + n_1k_1*L1Stage_B <= L1Size / 2Byte$
- $m_0k_0*L0AStage <= L0ASize / 2Byte$
- $n_0k_0*L0BStage <= L0BSize / 2Byte$
- $n_0n_0*L0CStage <= L0CSize / 4Byte$
- $m_0 = m_1$
- $n_0 = n_1$

### 读取数据量
每个基本任务块需要搬运$m_1K+Kn_1$的数据，总读取量为：

$2Byte * [m_1K+Kn_1] * \frac{MN}{m_1n_1} = 2Byte * MNK * [\frac{1}{m_1}+\frac{1}{n_1}]$

### 写出数据量
每个基本任务块计算得到$m_1n_1$的结果并搬出，总写出量为：

$2Byte * MN$

### 计算量
输出矩阵C的每个数据点需要$K$次乘加，总计算量固定为：

$2MNK$

计算耗时多数情况下为刚性时间，仅与参与计算的AIC核数相关，在各理论模板中都一样，后续不再赘述。
</details>

<details>
<summary><strong><font size="4">多核切K模板 MultiCoreSplitK</font></strong></summary>

### Tiling建模

<img src="https://raw.gitcode.com/user-images/assets/7801479/d4b0e2d2-4333-4df4-9af2-44654cc37e54/2multiCoreSplitK.png" width="80%">

如图展示一个常规的fp16的矩阵运算（L0C上按照fp32累加），$MN$方向共切分12个基本任务块，假设有24个AIC物理核，此时负载不均衡，故引入$K$轴切分成2个$k$、产生24个基本任务块，在AIC上负载均衡。定义相关参数：
- 问题shape：$M$，$N$，$K$
- 搬入L1Cache时的TileShape：$m_1$，$n_1$，$k_1$
- 搬入L0A/LOB、搬出L0C时的TileShape：$m_0$，$n_0$，$k_0$
- <font color="red">相较Common模板</font>，新增$K$方向切分长度$k$

相比`Common模板`，为了减少读取数据量，在较大的$m_1$、$n_1$下，可能存在负载均衡问题，即$MN$方向切分的任务块远少于AIC核数，导致读取带宽不高（核数不够），所以加入$K$方向分核。采用 $M$、$N$、$K$ 方向分核，产生$\frac{MNK}{m_1n_1k}$个基本任务块、分配给AIC核完成搬运和计算，每个基本任务块需要搬运$m_1k+kn_1$的数据、计算得到$m_1n_1$的结果并搬出。硬件上约束与Common相同。

### 读取数据量
每个基本任务块需要搬运$m_1K+Kn_1$的数据，总读取量与`Common模板`一致：

$2Byte * [m_1k+kn_1] * \frac{MNK}{m_1n_1k} = 2Byte * MNK * [\frac{1}{m_1}+\frac{1}{n_1}]$

### 写出数据量
每个基本任务块计算得到$m_1n_1$的结果并搬出，需要$\frac{K}{k}$个基本块累加来得到输出矩阵C的$m_1n_1$块的最终输出，总写出量为：

$2Byte * MNK / k$

### 定性分析

相较`Common模板`，搬入数据量不变，写出数据量增加，并产生后处理ReduceAdd的开销（包含AIV启动的开销），但切分基本块更多、更易负载均衡。

</details>

<details>
<summary><strong><font size="4">单核切K模板 SingleCoreSplitK</font></strong></summary>

### Tiling建模

<img src="https://raw.gitcode.com/user-images/assets/7801479/e16f5a39-2f7b-4a72-9d79-502cc8682e75/3singleCoreSplitK.png" width="80%">

如图展示一个常规的fp16的矩阵运算（L0C上按照fp32累加），定义相关参数：
- 问题shape：$M$，$N$，$K$
- 搬入L1Cache时的TileShape：$m_1$，$n_1$，$k_1$
- 搬入L0A/LOB、搬出L0C时的TileShape：$m_0$，$n_0$，$k_0$

相比`Common模板`，为了减少读取数据量，进一步增大抽象上的$m_1$、$n_1$，考虑将$m_1k_1$的tile块直接与对应的所有$k_1n_1$的tile块完成计算（等同于将$n_1$放大到$N$），此时输出$m_0n_0$的tile块没法在$L0C$常驻，需要及时搬出，通过`atomicAdd`在`GM`上累加。硬件上约束如下：
- $m_1k_1*L1Stage_A + n_1k_1*L1Stage_B <= L1Size / 2Byte$
- $m_0k_0*L0AStage <= L0ASize / 2Byte$
- $n_0k_0*L0BStage <= L0BSize / 2Byte$
- $n_0n_0*L0CStage <= L0CSize / 4Byte$
- $m_0 <= m_1$
- $n_0 <= n_1$

### 读取数据量
在`Common模板`的读取数据量公式上，将$n_1$放大到$N$；或者从A矩阵分基本任务块来理解，切分$\frac{MK}{m_1k_1}$个基本块，每个基本块完成搬入此块A矩阵tile块以及对应全部的B矩阵tile块、即搬入$m_1k_1+k_1N$的数据：

$2Byte * [m_1k_1+k_1N] * \frac{MK}{m_1k_1} = 2Byte * MNK * [\frac{1}{m_1}+\frac{1}{N}]$

### 写出数据量
切分A矩阵分基本任务块，共$\frac{MK}{m_1k_1}$个基本块，每个基本任务块计算得到$m_1N$的结果并搬出，总写出量为：

$2Byte * MNK / k_1$

### 定性分析

相较`Common模板`，搬入数据量减少，写出数据量增加，与AIV无关。

</details>

## 工程优化清单

<details>
<summary><strong><font size="4">流水优化（Preload）</font></strong></summary>

### 现象分析
### 优化方案

</details>

<details>
<summary><strong><font size="4">读取带宽优化（Padding）</font></strong></summary>

### 现象分析
### 优化方案
#### PaddingMatrixND
#### PaddingMatrixBlockND
#### PaddingMatrixNZ

</details>

<details>
<summary><strong><font size="4">读取带宽优化（ShuffleK）</font></strong></summary>

### 现象分析
### 优化方案

</details>

<details>
<summary><strong><font size="4">读取带宽优化（小M下指令替换）</font></strong></summary>

### 现象分析
### 优化方案

</details>

<details>
<summary><strong><font size="4">读取带宽优化（L1常驻）</font></strong></summary>

### 现象分析
### 优化方案

</details>

<details>
<summary><strong><font size="4">Scalar开销消减</font></strong></summary>

### 现象分析
### 优化方案

</details>

<details>
<summary><strong><font size="4">写出带宽优化</font></strong></summary>

### 现象分析
### 优化方案

</details>