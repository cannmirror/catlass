# CATLASS Python API 设计文档 - GEMM接口

## 概述

本文档描述了CATLASS（华为昇腾NPU上的CUTLASS实现）的Python接口设计。该设计对标NVIDIA CUTLASS的Python接口，同时针对昇腾硬件特点进行了简化和优化。

## 设计原则

1. **对标CUTLASS**: 接口风格和命名尽量与NVIDIA CUTLASS保持一致
2. **简化设计**: 隐藏底层模板复杂性，提供简洁易用的API
3. **面向对象**: 使用操作器类封装GEMM操作

## 核心模块

### 1. 数据类型 (DataType)

```python
from enum import Enum

class DataType(Enum):
    """支持的数据类型"""
    F16 = "float16"      # half precision
    F32 = "float32"      # single precision
    F64 = "float64"      # double precision (如果支持)
    I8 = "int8"          # 8-bit integer
    I16 = "int16"        # 16-bit integer
    I32 = "int32"        # 32-bit integer
    U8 = "uint8"         # unsigned 8-bit integer
    FP8_E4M3 = "fp8_e4m3"  # FP8 E4M3 format
    FP8_E5M2 = "fp8_e5m2"  # FP8 E5M2 format
```

### 2. 布局类型 (Layout)

```python
from enum import Enum

class Layout(Enum):
    """矩阵布局类型"""
    ROW_MAJOR = "row_major"        # C-style (row-major)
    COLUMN_MAJOR = "column_major"  # Fortran-style (column-major)
    ZN = "zn"                      # 昇腾特定布局
    NZ = "nz"                      # 昇腾特定布局
    # 其他昇腾特定布局可根据需要添加
```

### 3. 架构类型 (Architecture)

```python
from enum import Enum

class Architecture(Enum):
    """支持的硬件架构"""
    ATLAS_A2 = "atlas_a2"
    ATLAS_A3 = "atlas_a3"
    # 未来可扩展其他架构
```

### 4. GEMM操作

#### 4.1 基础GEMM接口

```python
def gemm(
    A: ArrayLike,
    B: ArrayLike,
    C: Optional[ArrayLike] = None,
    *,
    alpha: float = 1.0,
    beta: float = 0.0,
    layout_a: Layout = Layout.ROW_MAJOR,
    layout_b: Layout = Layout.ROW_MAJOR,
    layout_c: Layout = Layout.ROW_MAJOR,
    trans_a: bool = False,
    trans_b: bool = False,
    dtype: Optional[DataType] = None,
    device_id: int = 0,
    stream: Optional[Stream] = None
) -> ArrayLike:
    """
    通用矩阵乘法: C = alpha * op(A) * op(B) + beta * C
    
    Args:
        A: 输入矩阵A，形状为 (M, K) 或 (K, M) (如果trans_a=True)
        B: 输入矩阵B，形状为 (K, N) 或 (N, K) (如果trans_b=True)
        C: 输出矩阵C，形状为 (M, N)。如果为None，将自动分配
        alpha: 缩放因子，默认1.0
        beta: 缩放因子，默认0.0
        layout_a: 矩阵A的布局类型
        layout_b: 矩阵B的布局类型
        layout_c: 矩阵C的布局类型
        trans_a: 是否转置A
        trans_b: 是否转置B
        dtype: 输出数据类型，如果为None则从输入推断
        device_id: NPU设备ID
        stream: 计算流，如果为None则使用默认流
        
    Returns:
        结果矩阵C
        
    Example:
        >>> import numpy as np
        >>> A = np.random.randn(1024, 512).astype(np.float16)
        >>> B = np.random.randn(512, 256).astype(np.float16)
        >>> C = gemm(A, B)
    """
    pass
```

#### 4.2 优化的GEMM接口

```python
def optimized_gemm(
    A: ArrayLike,
    B: ArrayLike,
    C: Optional[ArrayLike] = None,
    *,
    alpha: float = 1.0,
    beta: float = 0.0,
    layout_a: Layout = Layout.ROW_MAJOR,
    layout_b: Layout = Layout.ROW_MAJOR,
    layout_c: Layout = Layout.ROW_MAJOR,
    trans_a: bool = False,
    trans_b: bool = False,
    dtype: Optional[DataType] = None,
    arch: Architecture = Architecture.ATLAS_A2,
    enable_padding: bool = True,
    enable_preload: bool = True,
    device_id: int = 0,
    stream: Optional[Stream] = None
) -> ArrayLike:
    """
    优化的GEMM操作，使用硬件特定的优化策略
    
    Args:
        A, B, C: 同gemm()
        alpha, beta: 同gemm()
        layout_a, layout_b, layout_c: 同gemm()
        trans_a, trans_b: 同gemm()
        dtype: 同gemm()
        arch: 目标硬件架构
        enable_padding: 是否启用padding优化
        enable_preload: 是否启用preload优化
        device_id: NPU设备ID
        stream: 计算流
        
    Returns:
        结果矩阵C
    """
    pass
```

#### 4.3 基础Matmul接口（简化版）

```python
def matmul(
    A: ArrayLike,
    B: ArrayLike,
    *,
    trans_a: bool = False,
    trans_b: bool = False,
    dtype: Optional[DataType] = None,
    device_id: int = 0,
    stream: Optional[Stream] = None
) -> ArrayLike:
    """
    简化的矩阵乘法接口: C = A @ B
    
    Args:
        A: 输入矩阵A
        B: 输入矩阵B
        trans_a: 是否转置A
        trans_b: 是否转置B
        dtype: 输出数据类型
        device_id: NPU设备ID
        stream: 计算流
        
    Returns:
        结果矩阵C = A @ B
    """
    return gemm(A, B, alpha=1.0, beta=0.0, trans_a=trans_a, trans_b=trans_b, 
                dtype=dtype, device_id=device_id, stream=stream)
```

### 5. Grouped GEMM操作

```python
def grouped_gemm(
    problem_shapes: List[Tuple[int, int, int]],  # List of (M, N, K)
    A_list: List[ArrayLike],
    B_list: List[ArrayLike],
    C_list: Optional[List[ArrayLike]] = None,
    *,
    layout_a: Layout = Layout.COLUMN_MAJOR,
    layout_b: Layout = Layout.ROW_MAJOR,
    layout_c: Layout = Layout.ROW_MAJOR,
    dtype: Optional[DataType] = None,
    device_id: int = 0,
    stream: Optional[Stream] = None
) -> List[ArrayLike]:
    """
    分组GEMM操作，同时执行多个不同形状的GEMM
    
    Args:
        problem_shapes: 每个GEMM问题的形状列表，每个元素为(M, N, K)
        A_list: 矩阵A的列表
        B_list: 矩阵B的列表
        C_list: 输出矩阵C的列表，如果为None则自动分配
        layout_a, layout_b, layout_c: 布局类型
        dtype: 输出数据类型
        device_id: NPU设备ID
        stream: 计算流
        
    Returns:
        结果矩阵C的列表
        
    Example:
        >>> shapes = [(128, 256, 512), (256, 128, 512)]
        >>> A_list = [np.random.randn(128, 512).astype(np.float16) for _ in shapes]
        >>> B_list = [np.random.randn(512, 256).astype(np.float16) for _ in shapes]
        >>> C_list = grouped_gemm(shapes, A_list, B_list)
    """
    pass
```

### 6. GEMV操作（矩阵-向量乘法）

```python
def gemv(
    A: ArrayLike,
    x: ArrayLike,
    y: Optional[ArrayLike] = None,
    *,
    alpha: float = 1.0,
    beta: float = 0.0,
    trans: bool = False,
    layout_a: Layout = Layout.ROW_MAJOR,
    dtype: Optional[DataType] = None,
    device_id: int = 0,
    stream: Optional[Stream] = None
) -> ArrayLike:
    """
    矩阵-向量乘法: y = alpha * op(A) * x + beta * y
    
    Args:
        A: 输入矩阵，形状为 (M, N)
        x: 输入向量，形状为 (N,) 或 (M,) (如果trans=True)
        y: 输出向量，形状为 (M,) 或 (N,) (如果trans=True)
        alpha: 缩放因子
        beta: 缩放因子
        trans: 是否转置A
        layout_a: 矩阵A的布局类型
        dtype: 输出数据类型
        device_id: NPU设备ID
        stream: 计算流
        
    Returns:
        结果向量y
    """
    pass
```

### 7. 卷积操作

```python
def conv2d(
    input: ArrayLike,
    weight: ArrayLike,
    bias: Optional[ArrayLike] = None,
    output: Optional[ArrayLike] = None,
    *,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1),
    groups: int = 1,
    dtype: Optional[DataType] = None,
    device_id: int = 0,
    stream: Optional[Stream] = None
) -> ArrayLike:
    """
    2D卷积操作
    
    Args:
        input: 输入张量，形状为 (N, C, H, W)
        weight: 卷积核，形状为 (C_out, C_in, K_h, K_w)
        bias: 偏置向量，形状为 (C_out,)，可选
        output: 输出张量，如果为None则自动分配
        stride: 步长 (stride_h, stride_w)
        padding: 填充 (pad_h, pad_w)
        dilation: 膨胀率 (dil_h, dil_w)
        groups: 分组数
        dtype: 输出数据类型
        device_id: NPU设备ID
        stream: 计算流
        
    Returns:
        输出张量
    """
    pass
```

### 8. 带激活函数的GEMM

```python
def gemm_with_activation(
    A: ArrayLike,
    B: ArrayLike,
    C: Optional[ArrayLike] = None,
    *,
    activation: str = "none",  # "none", "relu", "gelu", "swish"
    alpha: float = 1.0,
    beta: float = 0.0,
    layout_a: Layout = Layout.ROW_MAJOR,
    layout_b: Layout = Layout.ROW_MAJOR,
    layout_c: Layout = Layout.ROW_MAJOR,
    trans_a: bool = False,
    trans_b: bool = False,
    dtype: Optional[DataType] = None,
    device_id: int = 0,
    stream: Optional[Stream] = None
) -> ArrayLike:
    """
    带激活函数的GEMM: C = activation(alpha * op(A) * op(B) + beta * C)
    
    Args:
        A, B, C: 同gemm()
        activation: 激活函数类型
        alpha, beta: 同gemm()
        layout_a, layout_b, layout_c: 同gemm()
        trans_a, trans_b: 同gemm()
        dtype: 同gemm()
        device_id: NPU设备ID
        stream: 计算流
        
    Returns:
        结果矩阵C
    """
    pass
```

### 9. 带偏置的GEMM

```python
def gemm_with_bias(
    A: ArrayLike,
    B: ArrayLike,
    bias: ArrayLike,
    C: Optional[ArrayLike] = None,
    *,
    alpha: float = 1.0,
    beta: float = 0.0,
    layout_a: Layout = Layout.ROW_MAJOR,
    layout_b: Layout = Layout.ROW_MAJOR,
    layout_c: Layout = Layout.ROW_MAJOR,
    trans_a: bool = False,
    trans_b: bool = False,
    dtype: Optional[DataType] = None,
    device_id: int = 0,
    stream: Optional[Stream] = None
) -> ArrayLike:
    """
    带偏置的GEMM: C = alpha * op(A) * op(B) + bias + beta * C
    
    Args:
        A, B, C: 同gemm()
        bias: 偏置向量，形状为 (M,) 或广播兼容的形状
        alpha, beta: 同gemm()
        layout_a, layout_b, layout_c: 同gemm()
        trans_a, trans_b: 同gemm()
        dtype: 同gemm()
        device_id: NPU设备ID
        stream: 计算流
        
    Returns:
        结果矩阵C
    """
    pass
```

### 10. 量化GEMM

```python
def quantized_gemm(
    A: ArrayLike,  # int8 or int4
    B: ArrayLike,  # int8 or int4
    scales_a: Optional[ArrayLike] = None,
    scales_b: Optional[ArrayLike] = None,
    C: Optional[ArrayLike] = None,
    *,
    quant_type: str = "w8a16",  # "w8a16", "w4a8", etc.
    layout_a: Layout = Layout.ROW_MAJOR,
    layout_b: Layout = Layout.ROW_MAJOR,
    layout_c: Layout = Layout.ROW_MAJOR,
    dtype: DataType = DataType.F16,
    device_id: int = 0,
    stream: Optional[Stream] = None
) -> ArrayLike:
    """
    量化GEMM操作
    
    Args:
        A: 量化后的矩阵A
        B: 量化后的矩阵B
        scales_a: A的缩放因子
        scales_b: B的缩放因子
        C: 输出矩阵C
        quant_type: 量化类型
        layout_a, layout_b, layout_c: 布局类型
        dtype: 输出数据类型
        device_id: NPU设备ID
        stream: 计算流
        
    Returns:
        结果矩阵C
    """
    pass
```

### 11. Split-K GEMM

```python
def splitk_gemm(
    A: ArrayLike,
    B: ArrayLike,
    C: Optional[ArrayLike] = None,
    *,
    split_k: int = 1,
    alpha: float = 1.0,
    beta: float = 0.0,
    layout_a: Layout = Layout.ROW_MAJOR,
    layout_b: Layout = Layout.ROW_MAJOR,
    layout_c: Layout = Layout.ROW_MAJOR,
    trans_a: bool = False,
    trans_b: bool = False,
    dtype: Optional[DataType] = None,
    device_id: int = 0,
    stream: Optional[Stream] = None
) -> ArrayLike:
    """
    Split-K GEMM，将K维度分割以支持更大的矩阵
    
    Args:
        A, B, C: 同gemm()
        split_k: K维度的分割数
        alpha, beta: 同gemm()
        layout_a, layout_b, layout_c: 同gemm()
        trans_a, trans_b: 同gemm()
        dtype: 同gemm()
        device_id: NPU设备ID
        stream: 计算流
        
    Returns:
        结果矩阵C
    """
    pass
```

### 12. Flash Attention相关操作

```python
def flash_attention(
    Q: ArrayLike,
    K: ArrayLike,
    V: ArrayLike,
    O: Optional[ArrayLike] = None,
    *,
    head_dim: int,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    dtype: Optional[DataType] = None,
    device_id: int = 0,
    stream: Optional[Stream] = None
) -> ArrayLike:
    """
    Flash Attention推理操作
    
    Args:
        Q: Query矩阵
        K: Key矩阵
        V: Value矩阵
        O: 输出矩阵
        head_dim: 注意力头维度
        softmax_scale: Softmax缩放因子
        causal: 是否使用因果掩码
        dtype: 输出数据类型
        device_id: NPU设备ID
        stream: 计算流
        
    Returns:
        输出矩阵O
    """
    pass
```

## 高级接口

### 13. GEMM操作器类（面向对象接口）

```python
class GemmOperation:
    """GEMM操作的面向对象接口，支持配置复用"""
    
    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        *,
        dtype_a: DataType = DataType.F16,
        dtype_b: DataType = DataType.F16,
        dtype_c: DataType = DataType.F16,
        layout_a: Layout = Layout.ROW_MAJOR,
        layout_b: Layout = Layout.ROW_MAJOR,
        layout_c: Layout = Layout.ROW_MAJOR,
        arch: Architecture = Architecture.ATLAS_A2,
        device_id: int = 0
    ):
        """
        初始化GEMM操作
        
        Args:
            m, n, k: 矩阵维度
            dtype_a, dtype_b, dtype_c: 数据类型
            layout_a, layout_b, layout_c: 布局类型
            arch: 硬件架构
            device_id: NPU设备ID
        """
        pass
    
    def can_implement(self) -> bool:
        """检查当前配置是否可以被实现"""
        pass
    
    def get_workspace_size(self) -> int:
        """获取所需的工作空间大小（字节）"""
        pass
    
    def run(
        self,
        A: ArrayLike,
        B: ArrayLike,
        C: Optional[ArrayLike] = None,
        *,
        alpha: float = 1.0,
        beta: float = 0.0,
        workspace: Optional[ArrayLike] = None,
        stream: Optional[Stream] = None
    ) -> ArrayLike:
        """
        执行GEMM操作
        
        Args:
            A, B, C: 输入输出矩阵
            alpha, beta: 缩放因子
            workspace: 工作空间，如果为None则自动分配
            stream: 计算流
            
        Returns:
            结果矩阵C
        """
        pass
```

### 14. 配置和调优

```python
class GemmConfig:
    """GEMM配置类，用于性能调优"""
    
    def __init__(
        self,
        *,
        l1_tile_shape: Tuple[int, int, int] = (128, 256, 256),
        l0_tile_shape: Tuple[int, int, int] = (128, 256, 64),
        enable_unit_flag: bool = True,
        enable_shuffle_k: bool = True,
        enable_preload: bool = True,
        enable_padding: bool = True,
        swizzle_offset: int = 3,
        swizzle_direction: int = 0
    ):
        """
        初始化GEMM配置
        
        Args:
            l1_tile_shape: L1缓存tile形状 (M, N, K)
            l0_tile_shape: L0缓存tile形状 (M, N, K)
            enable_unit_flag: 是否启用unit flag优化
            enable_shuffle_k: 是否启用K维度shuffle
            enable_preload: 是否启用preload
            enable_padding: 是否启用padding
            swizzle_offset: Swizzle偏移量
            swizzle_direction: Swizzle方向
        """
        pass

def auto_tune_gemm(
    m: int,
    n: int,
    k: int,
    *,
    dtype: DataType = DataType.F16,
    layout_a: Layout = Layout.ROW_MAJOR,
    layout_b: Layout = Layout.ROW_MAJOR,
    layout_c: Layout = Layout.ROW_MAJOR,
    arch: Architecture = Architecture.ATLAS_A2,
    device_id: int = 0
) -> GemmConfig:
    """
    自动调优GEMM配置，返回最优配置
    
    Args:
        m, n, k: 矩阵维度
        dtype: 数据类型
        layout_a, layout_b, layout_c: 布局类型
        arch: 硬件架构
        device_id: NPU设备ID
        
    Returns:
        最优的GemmConfig
    """
    pass
```

## 工具函数

### 15. 内存管理

```python
class MemoryPool:
    """内存池，用于高效的内存管理"""
    
    def __init__(self, device_id: int = 0):
        """初始化内存池"""
        pass
    
    def allocate(self, size: int, dtype: DataType) -> ArrayLike:
        """从内存池分配内存"""
        pass
    
    def free(self, ptr: ArrayLike) -> None:
        """释放内存到内存池"""
        pass
    
    def reset(self) -> None:
        """重置内存池"""
        pass

def get_device_memory_info(device_id: int = 0) -> Dict[str, int]:
    """
    获取设备内存信息
    
    Returns:
        包含总内存和可用内存的字典
    """
    pass
```

### 16. 流管理

```python
class Stream:
    """计算流对象"""
    
    def __init__(self, device_id: int = 0):
        """创建新的计算流"""
        pass
    
    def synchronize(self) -> None:
        """同步流"""
        pass
    
    def __enter__(self):
        """上下文管理器支持"""
        pass
    
    def __exit__(self, *args):
        """上下文管理器支持"""
        pass

def get_default_stream(device_id: int = 0) -> Stream:
    """获取默认计算流"""
    pass
```

### 17. 设备管理

```python
def get_device_count() -> int:
    """获取可用NPU设备数量"""
    pass

def set_device(device_id: int) -> None:
    """设置当前设备"""
    pass

def get_device() -> int:
    """获取当前设备ID"""
    pass

def device_synchronize(device_id: int = 0) -> None:
    """同步设备"""
    pass
```

## 类型定义

```python
from typing import Union, List, Tuple, Optional, Dict
import numpy as np

# ArrayLike可以是numpy数组、torch tensor或其他兼容类型
ArrayLike = Union[np.ndarray, "torch.Tensor", "cupy.ndarray"]
```

## 使用示例

### 示例1: 基础矩阵乘法

```python
import catlass
import numpy as np

# 创建输入矩阵
A = np.random.randn(1024, 512).astype(np.float16)
B = np.random.randn(512, 256).astype(np.float16)

# 执行矩阵乘法
C = catlass.matmul(A, B)
```

### 示例2: 优化的GEMM

```python
import catlass
import numpy as np

A = np.random.randn(2048, 1024).astype(np.float16)
B = np.random.randn(1024, 512).astype(np.float16)

# 使用优化的GEMM
C = catlass.optimized_gemm(
    A, B,
    layout_a=catlass.Layout.ROW_MAJOR,
    layout_b=catlass.Layout.COLUMN_MAJOR,
    enable_padding=True,
    enable_preload=True
)
```

### 示例3: 带激活函数的GEMM

```python
import catlass
import numpy as np

A = np.random.randn(1024, 512).astype(np.float16)
B = np.random.randn(512, 256).astype(np.float16)

# GEMM + ReLU
C = catlass.gemm_with_activation(A, B, activation="relu")
```

### 示例4: 分组GEMM

```python
import catlass
import numpy as np

# 定义多个不同形状的GEMM问题
shapes = [(128, 256, 512), (256, 128, 512), (64, 128, 256)]
A_list = [np.random.randn(m, k).astype(np.float16) for m, _, k in shapes]
B_list = [np.random.randn(k, n).astype(np.float16) for _, n, k in shapes]

# 执行分组GEMM
C_list = catlass.grouped_gemm(shapes, A_list, B_list)
```

### 示例5: 使用操作器类

```python
import catlass
import numpy as np

# 创建GEMM操作器
gemm_op = catlass.GemmOperation(
    m=1024, n=512, k=256,
    dtype_a=catlass.DataType.F16,
    dtype_b=catlass.DataType.F16,
    dtype_c=catlass.DataType.F16
)

# 检查是否可以实现
if gemm_op.can_implement():
    # 准备输入
    A = np.random.randn(1024, 256).astype(np.float16)
    B = np.random.randn(256, 512).astype(np.float16)
    
    # 执行操作
    C = gemm_op.run(A, B, alpha=1.0, beta=0.0)
```

### 示例6: 自动调优

```python
import catlass

# 自动调优配置
config = catlass.auto_tune_gemm(
    m=2048, n=1024, k=512,
    dtype=catlass.DataType.F16,
    arch=catlass.Architecture.ATLAS_A2
)

print(f"最优L1 tile形状: {config.l1_tile_shape}")
print(f"最优L0 tile形状: {config.l0_tile_shape}")
```

## 实现注意事项

1. **数组兼容性**: 接口应支持numpy数组、PyTorch tensor、以及其他常见的数组类型
2. **内存管理**: 自动管理设备内存分配和释放，同时提供手动控制选项
3. **错误处理**: 提供清晰的错误信息和异常处理
4. **性能**: 最小化Python开销，关键路径使用C++实现
5. **文档**: 提供完整的API文档和使用示例

## 未来扩展

1. **更多量化类型**: 支持更多量化格式和混合精度
2. **动态形状**: 支持动态形状的GEMM操作
3. **批处理**: 支持批处理的GEMM操作
4. **分布式**: 支持多设备分布式GEMM
5. **性能分析**: 集成性能分析工具

## 与NVIDIA CUTLASS的对应关系

| CATLASS Python API | NVIDIA CUTLASS Python API | 说明 |
|-------------------|--------------------------|------|
| `gemm()` | `cutlass.gemm()` | 基础GEMM操作 |
| `Layout` | `cutlass.Layout` | 布局类型枚举 |
| `DataType` | `cutlass.DataType` | 数据类型枚举 |
| `GemmOperation` | `cutlass.GemmOperation` | GEMM操作器类 |
| `Architecture` | `cutlass.Architecture` | 硬件架构枚举 |

## 总结

本设计提供了一个简洁、易用且功能完整的Python接口，对标NVIDIA CUTLASS的同时针对昇腾硬件进行了优化。接口设计遵循Python最佳实践，提供了函数式和面向对象两种使用方式，满足不同用户的需求。

