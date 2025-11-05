# CATLASS

## 🔥 Latest News

- [2025/10] 发行版[v1.2.0](https://gitcode.com/cann/catlass/releases/v1.2.0)发布，新增[Matmul算子泛化](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/102_dynamic_optimized_matmul)等示例，快速上手请参阅[这里](docs/quickstart.md)
- [2025/09] CATLASS模板库正式开源

## 📌 简介

CATLASS(**CA**NN **T**emplates for **L**inear **A**lgebra **S**ubroutine**s**)，中文名为昇腾算子模板库，是一个聚焦于提供高性能矩阵乘类算子基础模板的代码库。  

通过抽象分层的方式将矩阵类算子代码模板化。算子计算逻辑可以进行白盒化组装，让算子代码可复用，可替换，可局部修改。针对昇腾硬件特点进行设计，可以支持复杂场景流水排布，如Flash Attention等算子。在上层代码逻辑共享的同时，可以支持底层硬件差异特化。

本代码仓为CATLASS联创代码仓。结合昇腾生态力量，共同设计研发算子模板，并提供典型算子的高性能实现代码样例。

## 新版本发布说明 1.2.0
 - 关键特性
   - 新增[Matmul泛化工程](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/102_dynamic_optimized_matmul)示例
     + 自动依照特征尺寸确定Tiling参数
     + 可在预设的算子模板中择优选取

 - 更多样例
    
    - Attention：[Flash Attention推理](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/23_flash_attention_infer)
    - 卷积类算子：[2D卷积](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/33_basic_conv2d)，[3D卷积](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/24_conv_bias)
    - 优化类Matmul：[A矩阵全加载Matmul](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/25_matmul_full_loadA)，[小矩阵优化](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/31_small_matmul)
    - 后处理类Matmul：[MatmulRelu](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/26_matmul_relu)，[MatmulGelu](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/27_matmul_gelu)，[MatmulSwish](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/28_matmul_swish)
    - 反量化Matmul：[FP8类型反量化Matmul](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/29_a2_fp8_e4m3_matmul)，[INT8类型反量化Matmul](https://gitcode.com/cann/catlass/tree/v1.2.0/examples/30_w8a16_matmul)

 - 工具支持
    - 新增[`msTuner`](https://gitcode.com/cann/catlass/tree/v1.2.0/tools/tuner)工具，实现Tiling自动寻优
 
    ...

请参阅[CHANGELOG](CHANGELOG.md)以查看当前及历史版本的详细更新内容。

## 📁 目录结构说明

```bash
catlass
├── cmake                     # cmake工程文件
├── docs                      # 文档存放目录
├── examples                  # kernel算子样例总目录
|   ├── 00_basic_matmul       # 单算子样例
|   |   ├── basic_matmul.cpp  # Host侧算子调用
|   |   ├── CMakeLists.txt
|   |   └── README.md         # 算子说明示例
|   ├── ...   
|   └── python_extension      # Python调用CATLASS算子
|                             # 的工程组件
├── include                   # 模板头文件集
|   ├── catlass               # 不同层级的算子实现逻辑
|   └── tla                   # 计算关联的基础数据结构
├── scripts                   # 编译脚本
|   └── build.sh              # 算子样例编译脚本
├── tests                     # 测试用例
└── tools                     # 相关工具
    └── tuner                 # Tiling自动寻优工具
```
以上为关键目录，详细目录参见[项目目录](.)。

## 💻 软硬件配套说明

CATLASS所需的软硬件环境依赖如下：

 - 昇腾产品：[Atlas A2训练/推理产品](https://www.hiascend.com/document/detail/zh/AscendFAQ/ProduTech/productform/hardwaredesc_0001.html)
 - CPU架构：`aarch64`/`x86_64`
 - 系统：Linux, openEuler
 - 软件依赖：
   + `gcc` >= 7.5, < 13.0
   + `cmake` >= 3.22
   + `python` >= 3.8, < 3.12

下述编译环境经测试支持CATLASS构建：

| 系统 | `gcc` | `cmake` | `python` | 
| ----- | --- | --- | --- |
| Ubuntu 22.04 | `9.3` | `3.22`  |  `3.10` | 
| openEuler 22.03 SP4 | `10.3` | `3.22`  |  `3.10` | 


不同CATLASS发行版可支持的硬件平台及所需的最低[CANN](https://www.hiascend.com/developer/download/community/result?module=cann)版本如下表：

| CATLASS社区版本 | 最低支持CANN包版本 | 支持昇腾产品 | 
| ----- | ----- | ---------- | 
| v1.2.0(当前)<br>~v1.0.0 | [8.2.RC1.alpha002](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.2.RC1.alpha002) | [Atlas A2训练/推理产品](https://www.hiascend.com/document/detail/zh/AscendFAQ/ProduTech/productform/hardwaredesc_0001.html) | 

- 对于某些调测工具，可能需要较上述更新的CANN版本，详请参考[调测工具文档](./docs/evaluation_collections.md)。

## ⚡️ 快速上手

为快速体验CATLASS的算子开发与使用，请参考下述内容。
 - [快速入门](./docs/quickstart.md)：以基础Matmul算子为例，基于CATLASS的第一个算子开发与编译；
 - [开发者实践](./docs/tutorials.md): 从算子编写至编译测试，再到Tiling调优与算子优化，从新手到进阶的实践示例。

## 📚 参考资料

参考下述资料可助力您开展CATLASS算子开发与测试实践，并深入算子调优逻辑，基于CATLASS实现更优性能的GEMM类算子。
 - [CATLASS API](./docs/api.md): 介绍CATLASS的分层特征与通用矩阵乘法Gemm API。
 - [CATLASS性能调测](./docs/evaluation_collections.md): 汇总CATLASS工程开发中的调测办法，有助于消除漏洞，分析性能的瓶颈点。
 - [CATLASS进阶实践](./docs/advanced_collections.md): 汇总CATLASS的进阶知识，如Tiling调参方法、Dispatch策略等，实现更高性能。

## 👥 合作贡献者

### [华南理工大学 陆璐教授团队](https://www2.scut.edu.cn/cs/2017/0629/c22284a328108/page.htm)

### 科大讯飞 研究院工程组

## 📝相关信息

- [贡献指南](CONTRIBUTING.md)
- [安全声明](SECURITYNOTE.md)
- [许可证](LICENSE)