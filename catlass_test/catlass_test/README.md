# CATLASS TEST

catlass_test是对catlass算子的python封装，不同于example中的pybind封装或者Python DSL，你可以用最少的代码接入并调起catlass算子，从而快速进行体验和测试。

## 项目结构

```bash
catlass_test/
├── catlass_test
│   ├── adapter     # 算子模板适配器
│   ├── catlass     # 用python实现的catlass数据结构
│   ├── common      # 公共函数
│   ├── compiler    # 算子模板编译器
│   ├── csrc        # examples的模板化代码
│   ├── golden      # 部分算子的golden实现
│   ├── interface   # 封装的函数接口
│   └── __init__.py # 入口文件
├── docs            # 文档
├── examples        # 快速接入示例
│   ├── custom_matmul_adapter.py
│   ├── custom_matmul.cpp
│   └── test_custom_matmul.py
├── pyproject.toml
├── README.md
└── tests           # 测试用例

```

## 快速体验catlass_test

### 用catlass_test调用已接入的examples算子

```py
import torch
import catlass_test
a = torch.ones([4, 4], dtype=torch.float16).npu()
b = torch.ones([4, 4], dtype=torch.float16).npu()
# catlass_test interface calling
c_catlass_test = catlass_test.basic_matmul(a, b)
print(c_catlass_test)
>>> tensor([[4., 4., 4., 4.],
        [4., 4., 4., 4.],
        [4., 4., 4., 4.],
        [4., 4., 4., 4.]], device='npu:0', dtype=torch.float16)
```

### 将自定义算子`custom_matmul`接入catlass_test

```cpp
// custom_matmul.hpp
template <ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC>
inline int32_t CustomMatmul(GemmCoord problemShape, uint8_t *deviceA, uint8_t *deviceB, uint8_t *deviceC, aclrtStream stream){
    // catlass API combinations
}
```

```py
# custom_matmul.py
import torch
from catlass_test.adapter import MatmulAdapter
def custom_matmul(input: torch.Tensor, mat2: torch.Tensor)->torch.Tensor:
    case = MatmulCase(
        "custom_matmul.hpp"
        {"A": input, "B": mat2},
    )
    case.run()
    return case.get_tensor("C")
a = torch.ones([4, 4], dtype=torch.float16).npu()
b = torch.ones([4, 4], dtype=torch.float16).npu()
c = custom_matmul(a, b)
print(c)
>>> tensor([[4., 4., 4., 4.],
        [4., 4., 4., 4.],
        [4., 4., 4., 4.],
        [4., 4., 4., 4.]], device='npu:0', dtype=torch.float16)
```

## example支持范围

|示例名|特性|是否支持|
|---|---|---|
| `00_basic_matmul` | mm |✅|
| `01_batched_matmul` |  bmm ||
| `02_grouped_matmul_slice_m` |  gmm ||
| `03_matmul_add` | mm+add ||
| `04_padding_matmul` |  mm |✅|
| `05_grouped_matmul_slice_k` |  gmm ||
| `06_optimized_matmul` |  mm ||
| `07_grouped_matmul_slice_m_per_token_dequant_moe` | gmm+dequant ||
| `08_grouped_matmul` |  gmm ||
| `09_splitk_matmul` |  mm |✅|
| `10_grouped_matmul_slice_m_per_token_dequant` |  gmm+dequant ||
| `11_grouped_matmul_slice_k_per_token_dequant` |  gmm+dequant ||
| `12_quant_matmul` | mm+dequant ||
| `13_basic_matmul_tla` |  mm ||
| `14_optimized_matmul_tla` |  mm ||
| `15_gemm` |  gemm ||
| `16_group_gemm` |  group_gemm ||
| `17_gemv_aiv` | gemm ||
| `18_gemv_aic` |  gemm ||
| `19_mla` | attn ||
| `20_matmul_bias` |  mm+add ||
| `21_basic_matmul_preload_zN` |  mm ||
| `22_padding_splitk_matmul` |  mm ||
| `23_flash_attention_infer` |  attn ||
| `24_conv_bias` |  conv ||
| `25_matmul_full_loadA` | mm ||
| `26_matmul_relu` | mm ||
| `27_matmul_gelu` |  mm ||
| `28_matmul_swish` |  mm ||
| `29_a2_fp8_e4m3_matmul` |  mm ||
| `30_w8a16_matmul` | mm ||
| `31_small_matmul` |  mm ||
