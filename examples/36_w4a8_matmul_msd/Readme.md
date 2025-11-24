# SingleSplitK_Matmul Example Readme
## 代码组织
```
├── 34_single_splitk_matmul
│   ├── CMakeLists.txt # CMake编译文件
│   ├── single_core_splitk.cpp 
│   └── README.md
```
## 功能介绍
- 提供了单核切K的Matmul实现
- 可依据对齐情况对A，B矩阵进行Padding操作

## 使用示例
- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/quickstart.md#算子编译)   

- 执行算子
```
# 编译指定用例
bash scripts/build.sh 34_single_core_splitk_matmul
cd output/bin
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID
# Device ID可选，默认为0
./34_single_core_splitk_matmul 256 512 1024 0
```
执行结果如下，说明精度比对成功。
```
Compare success.
```