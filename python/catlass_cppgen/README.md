# catlass_cppgen

## 项目介绍

通过python框架构建CATLASS算子并执行

## 使用样例

```py
import torch
from catlass_cppgen.op import Gemm
from catlass_cppgen.datatype import DataType
from catlass_cppgen.layout import LayoutEnum

# Step 1. 指定数据类型与布局，构建Gemm对象
gemm_plan = Gemm(element = DataType.FLOAT16, layout = LayoutEnum.RowMajor)
# 生成数据
a = torch.ones((16, 32), dtype = torch.float16, device='npu')
b = torch.ones((32, 64), dtype = torch.float16, device='npu')
# Step 2. 根据Tensor信息，决定使用Kernel
gemm_kernel = gemm_plan.get_kernel(A = a, B = b)
# Step 3. 编译Kernel，目前仅支持代码生成
if gemm_kernel is not None:
    gemm_kernel_src = gemm_kernel.codegen()
# call some compiler and run
# Step 4. 运行完Kernel,　修改Tiling,　尝试改善性能
gemm_kernel.tune(l1_tile_shape = GemmShape(128, 256, 192), l0_tile_shape = GemmShape(128, 64, 96))
gemm_kernel_src_tuned = gemm_kernel.codegen()
# call some compiler and run (tuned)

```
