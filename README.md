# CATLASS

## 🔥 Latest News

<!-- 更新 -->
- [2025/10] 发行版[v1.2.0](https://gitcode.com/cann/catlass/releases/v1.2.0)发布，新增[Matmul算子泛化](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/102_dynamic_optimized_matmul)等示例，快速上手请参阅[这里](docs/quickstart.md)
- [2025/09] CATLASS模板库正式开源

## 📌 简介

CATLASS(**CA**NN **T**emplates for **L**inear **A**lgebra **S**ubroutine**s**)，中文名为昇腾算子模板库，是一个聚焦于提供高性能矩阵乘类算子基础模板的代码库。  

通过抽象分层的方式将矩阵类算子代码模板化。算子计算逻辑可以进行白盒化组装，让算子代码可复用，可替换，可局部修改。针对昇腾硬件特点进行设计，可以支持复杂场景流水排布，如Flash Attention等算子。在上层代码逻辑共享的同时，可以支持底层硬件差异特化。

本代码仓为CATLASS联创代码仓。结合昇腾生态力量，共同设计研发算子模板，并提供典型算子的高性能实现代码样例。

## 新版本发布说明 1.2.0
 - 关键特性
   - 算子编译时支持传入计算平台架构(如编译选项`-DCATLASS_BISHENG_ARCH=a2`)<span>？是否有其他可选，如a3?</span>
   - 新增[Matmul泛化工程](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/102_dynamic_optimized_matmul)示例
     + 自动依照特征尺寸确定Tiling参数
     + 可在预设的算子模板中择优选取

   - 更新[Python调用接口](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/python_extension/README.md)内容
     + 调整工程组织结构
     + 支持转置情形

 - 更多样例
    - [Flash Attention推理算子](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/23_flash_attention_infer)
    - [2D卷积算子](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/33_basic_conv2d)
    - [3D卷积算子](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/24_conv_bias)
    - [A矩阵全加载Matmul算子](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/25_matmul_full_loadA)
    - [小矩阵优化Matmul算子](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/31_small_matmul)
    - [MatmulRelu算子](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/26_matmul_relu)
    - [MatmulGelu算子](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/27_matmul_gelu)
    - [MatmulSwish算子](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/28_matmul_swish)
    - [FP8类型反量化Matmul算子](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/29_a2_fp8_e4m3_matmul)
    - [INT8类型反量化Matmul算子](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/30_w8a16_matmul)

 - 工具支持
    - 新增[`msTuner`](https://gitcode.com/cann/catlass/tree/v1.2.0/tools/tuner)工具，用于Tiling自动寻优，在搜索空间内全量运行并获取性能数据
    - 支持使能[`msSanitizer`](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/devaids/optool/atlasopdev_16_0039.html)地址消毒工具(编译选项加入`--enable_mssanitizer`)

 - Bugfix与优化
   - 优化[`OptimizedMatmul`](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/06_optimized_matmul)算子实现，支持任意Padding方式组合
   - 修复`ASCEND_RT_VISIBLE_DEVICES`环境变量使能下，`msTuner`工具无法取得实际运行`DeviceId`的问题
   - 修复[PFA算子样例](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/19_mla)在单行数据场景下`Set/Wait`错配的异常情形
   - 修复[`OptimizedMatmul`](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/06_optimized_matmul)算子样例计算`Workspace`大小出错
   - 隔离使能`AscendC::Dump`及`AscendC::print`功能的代码段
   - 修复[`GroupedMatmulSliceK`](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/05_grouped_matmul_slice_k)算子在Ki=0特例时的输出清零行为，并将真值比较逻辑调整为全尺寸(M,N,K)比较
   - 修改[`performance_tools.md`](https://gitcode.com/cann/catlass/tree/v1.2.0/docs/tools/performance_tools.md)，[`tutorials.md`](https://gitcode.com/cann/catlass/tree/v1.2.0/docs/tutorials.md)等文档中的错误

请参阅[CHANGELOG](CHANGELOG.md)以取得历史版本的更新信息。



## 📁 目录结构说明

<!-- 目录结构补充齐全 -->
```bash
catlass
├── cmake          # cmake工程文件
├── docs           # 文档
├── examples       # kernel算子样例
├── include        # 模板头文件
├── scripts        # 编译脚本
|   └── build.sh   # 算子样例编译脚本
├── tests          # 测试用例
└── tools          # 相关工具
```

## 💻 软硬件配套说明

### 平台版本

在不同系统平台下，下述编译环境经测试可支持CATLASS构建：

| 系统 | `gcc` | `cmake` | `python` | 
| ----- | --- | --- | --- |
| Ubuntu 22.04 | `7.5`, `8.3`，`9.3`，`11.4` | `3.22`  |  `3.10` | 
| Ubuntu 20.04 | `7.5`, `8.3`，`9.3`，`11.4` | `3.22` | `3.10` | 
| Ubuntu 18.04 | `7.5`, `8.3`，`9.3`，`11.4` | `3.22` | `3.10` | 
| openEuler 22.03 | `7.3`, `10.3` | `3.22`  |  `3.10` | 

备注:
   - Catlass继承自CANN能力，支持`aarch64`/`x86_64`架构
   - 推荐使用`9.3`以上，`13.0`以下的GCC版本

### 版本匹配关系

CANN包赋能下，CATLASS能够在[昇腾系列AI处理器](https://www.hiascend.com/document/detail/zh/AscendFAQ/ProduTech/productform/hardwaredesc_0001.html)上运行，不同版本CATLASS可支持的硬件平台及其最低CANN包版本如下表：

| CATLASS社区版本 | 最低支持CANN包版本 | 支持昇腾产品 | 
| ----- | ----- | ---------- | 
| v1.2.0 | | `Atlas A2训练/推理产品` |
| v1.1.0 | | `Atlas A2训练/推理产品` |
| v1.0.0 | [8.2.RC1.alpha002](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.2.RC1.alpha002) | `Atlas A2训练/推理产品` |

备注：
- 对于某些调测工具，可能需要较以上版本更加新的CANN版本，可参考[调测工具文档](#toolbox)。

## ⚡️ 快速上手

为快速体验CATLASS的算子开发与使用，请参考下述资料。
 - [快速入门](./docs/quickstart.md)：以基础Matmul算子为例，演示基于CATLASS的开发与编译过程；
 - [GEMM API](./docs/api.md)：CATLASS的分层特征与通用矩阵乘法Gemm API。

## 📚 文档介绍
<!-- 合并下沉至docs/下：API介绍 基础文档 进阶文档 调测工具 -->

您可以查看
### 📖 基础文档

按照由浅入深的次序，对模板库的相关内容展开介绍。

- [quickstart](./docs/quickstart.md) - 快速上手实践模板库，以基础的Matmul算子开发为实践背景认识使用模板库。
- [catlass_optimize_guidance](./docs/catlass_optimize_guidance.md) - 模板库的进阶教程，介绍模板库下的基础调优方式，如何通过Tiling调参、应用不同的Dispatch策略的方式，快速获得性能提升。
- [api](./docs/api.md) - 介绍CATLASS模板库的通用矩阵乘法Gemm API。
- [swizzle_explanation](./docs/swizzle_explanation.md) - 对模板库中Swizzle策略的基本介绍，这影响了AI Core上计算基本块间的顺序。
- [dispatch_policies](./docs/dispatch_policies.md) - 对模板库在`Block`层面上`BlockMmad`中的一个重要模板参数`DispatchPolicy`的介绍。

### 🧰 调测工具文档

我们已经在CATLASS示例工程中适配了大多数CANN提供的调测工具，开发算子时，可基于CATLASS示例工程进行初步开发调优，无需关注具体的工具适配操作，待算子基础功能、性能达到预期，再迁移到其他工程中。

#### 🚗 功能调试

- [msDebug](./docs/tools/msdebug.md) - 类gdb/lldb的调试工具msDebug
  - ⚠️ **注意** 此功能依赖社区版`CANN`包版本为[8.2.RC1.alpha003](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.2.RC1.alpha003)。
- [printf](./docs/tools/print.md) - 在算子device代码进行打印调试
  - ⚠️ **注意** 此功能依赖社区版`CANN`包版本在CANN 8.3后（如[8.3.RC1.alpha001](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.3.RC1.alpha001)）。
- [ascendc_dump](./docs/tools/ascendc_dump.md) - 利用AscendC原生API进行调测

#### ✈️ 性能调优

- [msProf&Profiling](./docs/tools/performance_tools.md) - 性能调优工具`msProf`和`Profiling`
  - [单算子性能分析：msProf](./docs/tools/performance_tools.md#用msProf进行单算子性能分析)
  - [整网性能分析：Profiling](./docs/tools/performance_tools.md#用Profiling进行整网性能分析)
- [msTuner_CATLASS](./tools/tuner/README.md) - Tiling自动寻优工具

## 👥 合作贡献者

### [华南理工大学 陆璐教授团队](https://www2.scut.edu.cn/cs/2017/0629/c22284a328108/page.htm)

### 科大讯飞 研究院工程组

## 📝相关信息

- [贡献指南](CONTRIBUTING.md)
- [安全声明](SECURITYNOTE.md)
- [许可证](LICENSE)