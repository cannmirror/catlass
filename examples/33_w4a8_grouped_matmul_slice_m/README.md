# W4A8_Grouped_Matmul_Slice_M Example Readme
## 代码组织
```
├── 33_w4a8_grouped_matmul_slice_m
│   ├── CMakeLists.txt # CMake编译文件
│   ├── gen_data.py
│   ├── w4a8_grouped_matmul_slice_m.cpp 
│   └── README.md
```
## 功能介绍
- 该算子支持W4A8量化模式下A矩阵在m轴切分，然后和B矩阵按照group分组进行矩阵乘。
- A矩阵int8类型，shape为(m, n)，B矩阵int4类型，shape为(g, k, n)。每个group都把B矩阵cast成int8类型，矩阵乘法结果int32，然后通过不同scale的perTensor量化得到shape为(m, n)的half类型输出。
## 使用示例
- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/quickstart.md#算子编译)   

- 接下来，先执行`gen_data.py`，生成测试样例，测试用例需要从命令行输入, 执行该命令后会在当前路径下生成data目录，包含算子的输入数据和用于精度验证的golden数据。   
- 然后执行算子，这里要注意的是执行算子的输入shape和上面第一步生成数据的shape一致。

以下是一个完整的shell脚本示例
```
g=1221
m=860
k=5712
n=4535
device=0

function build() {
    rm -rf build
    rm -rf output
    bash ../../scripts/build.sh 33_w4a8_grouped_matmul_slice_m
}

function gen_data() {
    python3 gen_data.py $g $m $n $k
    echo "Data gen finished"
}

function run_kernel {
    echo 'Case: m=' $m ' g=' $g ' k=' $k ' n=' $n
    cd ../../output/bin/
    ./33_w4a8_grouped_matmul_slice_m $g $m $n $k $device
}

build
gen_data
run_kernel
```

执行结果如下，说明精度比对成功。
```
Compare success.
```