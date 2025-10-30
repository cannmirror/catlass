# msopgen example
aclnn接口是CANN软件栈一直沿用的接口，CANN提供msOpgen工具可以生成该接口的工程，便于用户接入自定义算子。该样例提供catlass算子接入msOpGen工程的示例代码与注意事项，并提供catlass风格的example调用示例。

以basic_matmul接入为例进行示例
## 编写json
参考xxx链接进行编写；库上提供[catlass_basic_matmul.json](examples/103_msopgen_aclnn_catlass_basic_matmul/catlass_basic_matmul.json)作为示例。
## 编写Host代码

参考[Host侧Tiling实现-基本流程](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/opdevg/Ascendcopdevg/atlas_ascendc_10_00021.html)实现`TilingFunc`。

若需要使能**算子入图**，请参考[算子入图（GE）图开发](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/opdevg/Ascendcopdevg/atlas_ascendc_10_0078.html)实现`InferShape`和`InferDataType`。

相关示例代码：

## 编写Device代码
参考[Kernel侧算子实现](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/opdevg/Ascendcopdevg/atlas_ascendc_10_0063.html)，实现kernel代码。

相关示例代码：

参考注意事项：

由于需要引入catlass的头文件，我们需要增加编译选项。在`op_kernel/CMakeList.txt`中增加算子编译选项：
```diff
# set custom compile options
if ("${CMAKE_BUILD_TYPE}x" STREQUAL "Debugx")
    add_ops_compile_options(ALL OPTIONS -g -O0)
endif()

+ add_ops_compile_options(ALL OPTIONS -I${CATLASS_INCLUDE_PATH})

add_kernels_compile()
```
其中`CATLASS_INCLUDE_PATH`是catlass代码仓下的include文件夹的路径。

msopgen的分离编译模式不支持直接将结构体（如`Catlass::GemmCoord`）。当需要使用结构体时，需要通过tiling data传递数据，然后在kernel侧重新构造。
```cpp
// 正确
extern "C" __global__ __aicore__ void
catlass_basic_matmul(GM_ADDR self, GM_ADDR mat2, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    Catlass::GemmCoord problemShape{tiling_data.m, tiling_data.n, tiling_data.k};
    // ...
}
// 暂不支持
extern "C" __global__ __aicore__ void
catlass_basic_matmul(GM_ADDR self, GM_ADDR mat2, GM_ADDR out, GM_ADDR workspace,  Catlass::GemmCoord problemShape)
{
    // ...
}
## 打包
。、build。sh
## 安装

# 编译指定用例
bash scripts/build.sh 00_basic_matmul
cd output/bin
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID
# Device ID可选，默认为0
./00_basic_matmul 256 512 1024 0
```
执行结果如下，说明精度比对成功。
```
Compare success.
```