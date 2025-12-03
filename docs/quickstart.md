# 模板库快速上手

## 环境准备

> **说明**：请先行确认[基础依赖](../README.md#-软硬件配套说明)、[NPU驱动](基础依赖、NPU驱动和固件已安装。)和固件已安装，及[基础依赖](../README.md#-软硬件配套说明)是否满足。

1. **安装社区版CANN toolkit包**

根据您所使用的[昇腾产品](https://www.hiascend.com/document/detail/zh/AscendFAQ/ProduTech/productform/hardwaredesc_0001.html)类别，请下载对应的CANN开发套件包`Ascend-cann-toolkit_{version}_linux-{arch}.run`，下载链接见[CANN toolkit](https://www.hiascend.com/zh/developer/download/community/result?module=cann)（有关CATLASS的版本支持情况详见[软件硬件配套说明](../README.md#软件硬件配套说明)）。

随后安装CANN开发套件包（详情参考[CANN安装指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/softwareinst/instg/instg_0008.html?Mode=PmIns&InstallType=local&OS=openEuler&Software=cannToolKit)）。

```bash
# 确保安装包有可执行权限
chmod +x Ascend-cann-toolkit_{version}_linux-{arch}.run
# 安装CANN toolkit包
./Ascend-cann-toolkit_{version}_linux-{arch}.run --full --force --install-path=${install_path}
```

- `{version}`: CANN包版本号。
- `{arch}`: 系统架构。
- `{install_path}`: 指定安装路径，默认为`/usr/local/Ascend`

2. **使能CANN 环境**

安装完成后，执行下述指令即完成CANN环境使能。

```bash
# 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}）
source /usr/local/Ascend/set_env.sh
# 指定路径安装
# source ${install_path}/set_env.sh
```

3. **下载源码**

本示例主要展示如何基于CATLASS快速搭建一个NPU上的BasicMatmul实现。示例中使用已提供的下层基础组件完成Device层和Kernel层组装，并调用算子输出结果。CATLASS分层示意图见[api文档](api.md)。

### Kernel层算子定义

Kernel层模板由Block层组件构成。这里首先定义三个Block层组件。
`<class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>`。

1. `BlockMmad_`为Block层mmad计算接口，定义方式如下：

```c++
using DispatchPolicy = Catlass::Gemm::MmadAtlasA2Pingpong<true>; //流水排布使用
using L1TileShape = Catlass::GemmShape<128, 256, 256>; // L1基本块
using L0TileShape = Catlass::GemmShape<128, 256, 64>; // L0基本块
using AType = Catlass::Gemm::GemmType<ElementA, LayoutA>;     //封装了A矩阵的数据类型和排布信息
using BType = Catlass::Gemm::GemmType<ElementB, LayoutB>;     //封装了B矩阵的数据类型和排布信息
using CType = Catlass::Gemm::GemmType<ElementC, LayoutC>;     //封装了C矩阵的数据类型和排布信息

using BlockMmad = Catlass::Gemm::Block::BlockMmad<DispatchPolicy,
    L1TileShape,
    L0TileShape,
    AType,
    BType,
    CType>;
```

2. `BlockEpilogue_`为Block层后处理，本文构建基础matmul，不涉及后处理，这里传入void。

```c++
using BlockEpilogue = void;
```

3. `BlockScheduler_`该模板类定义数据走位方式，提供计算offset的方法。此处使用定义好的GemmIdentityBlockSwizzle。参考[Swizzle策略说明](swizzle_explanation.md)文档了解更多swizzle信息。

```c++
using BlockScheduler = typename Catlass::Gemm::Block::GemmIdentityBlockSwizzle<>;
```

4. 基于上述组件即可完成BasicMatmul示例的Kernel层组装。

```c++
using MatmulKernel = Catlass::Gemm::Kernel::BasicMatmul<BlockMmad, BlockEpilogue, BlockScheduler>;
```

### Device层算子定义

基于Kernel层组装的算子，完成核函数的编写。

1. 使用CATLASS_GLOBAL修饰符定义Matmul函数，并传入算子的类型参数。

```c++
template <
    class LayoutA,
    class LayoutB,
    class LayoutC
>
CATLASS_GLOBAL
void BasicMatmul(
    GemmCoord problemShape,
    GM_ADDR gmA, LayoutA layoutA,
    GM_ADDR gmB, LayoutB layoutB,
    GM_ADDR gmC, LayoutC layoutC);
```

2. BasicMatmul的调用接口为`()`运算符，需要传入Params作为参数。

```c++
typename MatmulKernel::Params params{problemShape, gmA, layoutA, gmB, layoutB, gmC, layoutC};
```

3. 最后，实例化一个kernel，并执行该算子。

```c++
MatmulKernel matmul;
matmul(params);
```

### 算子调用

调用算子我们需要指定矩阵的输入输出的数据类型和数据排布信息，并使用`<<<>>>`的方式调用核函数。

```c++
BasicMatmul<<<BLOCK_NUM, nullptr, stream>>>(
        options.problemShape, deviceA, layoutA, deviceB, layoutB, deviceC, layoutC);
```

### 算子编译

- 设置源代码对应的编译语言，可以混合使用纯C++代码和算子代码。对于算子文件，需要设定语言为ASCEND。
- 调用`catlass_example_add_executable`函数指定target名称和编译文件。
  - `00_basic_matmul`为target名称
  - `basic_matmul.cpp`为需要编译的文件
  - `cube`为算子类型，可填写`cube`/`vec`/`mix`

```
# CMakeLists.txt
set_source_files_properties(basic_matmul.cpp PROPERTIES LANGUAGE ASC)
catlass_example_add_executable(
    00_basic_matmul
    cube
    basic_matmul.cpp
)
```

在项目目录下，调用`build.sh`，即可编译examples中的kernel代码。

```bash
# 下载项目源码，以master分支为例
git clone https://gitcode.com/cann/catlass.git
```

## 编译执行

> 模板库提供了一套可复用的模板、基础组件，赋能矩阵乘法算子开发，算子样例可见[样例目录](../examples/)。

1. **样例编译**

进入项目根目录，可执行下述编译指令：
```bash
bash scripts/build.sh [options] <target>
```

 - `options`: 可选编译选项，当前支持的选项包括：
   - `--clean`: 清理此前编译及输出目录（默认路径分别为`/build`，`/output`）。
   - `--debug`: 以Debug模式进行编译。
   - `--msdebug`：使能[msDebug](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/devaids/optool/atlasopdev_16_0062.html)工具，详见[在CATLASS样例工程使用msDebug](./tools/msdebug.md)。
   - `--simulator`：启用仿真器模式，该选项启用后将不在实际NPU上执行，详见[CATLASS样例仿真](./tools/performance_tools.md#msprof-op-simulator使用示例)。
   - `--enable_profiling`：使能Profiling工具，详见[CATLASS样例性能调优](./tools/performance_tools.md#profiling简介)。
   - `--enable_print`：启用编译器的打印功能，详见[基于`cce::printf`进行设备侧打印](./tools/print.md)。
   - `--enable_ascendc_dump`：启用`AscendC`相关算子调测API，详见[CATLASS样例使用AscendC算子调测API](./tools/ascendc_dump.md)。
   - `-DCATLASS_BISHENG_ARCH`：指明NPU架构，支持`a2`或`a3`。
   - `-D<option>`：给CMake传递其他的编译选项。

 - `target`： 要编译的算子样例，可指定为特定的样例名，也可指定为：
   - `catlass_examples`：对当前仓样例进行全量编译。
   - `python_extension`：编译pybind扩展，详见基于[Python调用CATLASS样例](../examples/python_extension/README.md)。
   - `torch_library`：编译torch扩展，详见[Python调用CATLASS样例](../examples/python_extension/README.md)。
   - `mstuner_catlass`：编译msTuner_CATLASS工具，详见[`mstuner_catlass`使用说明](../tools/tuner/README.md)。

以[basic_matmul](../examples/00_basic_matmul/README.md)样例编译过程为例，执行下述指令：
```bash
# 编译算子组件
bash scripts/build.sh 00_basic_matmul
```
若有下述提示信息，则编译成功。
```bash 
"[INFO] Target '{target}' built successfully."
```

2. **算子执行**

算子编译产物在`output/bin`路径下，切换至该目录下可运行算子样例程序。
以[basic_matmul]样例为例，可通过下述指令执行该算子：
```bash
# 切换至编译产物目录
cd output/bin
# ./00_basic_matmul m n k [deviceId]
./00_basic_matmul 256 512 1024 0
```

 - `256`， `512`， `1024`分别为矩阵乘法的在m轴、n轴、k轴的维度（左/右矩阵数据随机生成）
 - `deviceId`可选（默认为0），指定NPU卡的ID号。

执行该算子样例后，如出现下述结果则表明其计算符合精度预期。
```
Compare success.
```

请进一步参考[快速入门](./dev_guide.md#matmul算子开发)以开始第一个算子开发。