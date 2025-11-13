# EVG 使用参考

## 简介

**EVG（Epilogue Visitor Graph）** 是 [CATLASS](https://gitcode.com/cann/catlass) 中用于 GEMM 后处理（Epilogue）的声明式框架，后处理操作中的常见环节（如加法计算、类型转换、广播等）被抽象为为可组合的模板节点，并进一步以树形或拓扑结构拼接形成计算图，实现不同类型的后处理操作。
EVG支持图和节点复用与灵活扩展，开发者只需用"表达式"声明计算逻辑即可基于EVG框架进行算子后处理开发，数据搬运、空间分配、流水调度等具体细节均由框架自动处理。

## 快速上手

您可以参考[EVG快速入门](./evg_quickstart.md)，以一个[Matmul+Add](../../examples/32_matmul_add_evg/README.md)的后处理开发为起点开展EVG开发实践。

代码仓提供了下述样例可供参考：
 - [Matmul+Add](../../examples/32_matmul_add_evg/README.md)
 - [Matmul+Cast](../../examples/33_cast_evg/README.md)
 - [Matmul+Quant](../../examples/12_quant_matmul_evg/README.md)


## 进阶参考

如果您希望进一步了解EVG的进阶内容，实现更为复杂的算子后处理，请参考下述材料：
 - [EVT样例：QuantMatmul实现](./quant_matmul_cases_tutorial.md)
 - [EVG简要设计文档](./evg.md)
 - [API参考](./evg_api.md)
