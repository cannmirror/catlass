# W4A8_Grouped_Matmul_MSD Example Readme
## 代码组织
```
├── 36_w4a8_grouped_matmul_msd_full_loadA
│   ├── CMakeLists.txt # CMake编译文件
│   ├── gen_data.py
│   ├── w4a8_grouped_matmul_msd.cpp
│   └── README.md
```
## 功能介绍
- 提供了W4A8量化模式下grouped matmul的实现，采用MSD方案，可参考[极致优化GroupedMatmul量化方案](https://www.hiascend.com/developer/techArticles/20250717-2?envFlag=1)
- A矩阵行主序int8类型，B矩阵行主序int4类型；根据int8的高低4位bit把A矩阵拆成两个int4矩阵，并和B矩阵进行矩阵乘得到int32结果，通过随路转换实现per channel和per group量化并得到float16结果；通过vector计算进行精度补齐和per token量化，最终得到bfloat16输出类型
## 使用示例
- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/quickstart.md#算子编译)   

- 接下来，先执行`gen_data.py`，生成测试样例，测试用例需要从命令行输入, 执行该命令后会在当前路径下生成data目录，包含算子的输入数据和用于精度验证的golden数据。   
- 然后执行算子，这里要注意的是执行算子的输入shape和上面第一步生成数据的shape一致。

以下是一个完整的shell脚本示例
```
g=30
kGroupSize=320
m=77
k=1280
n=4536
device=0

function build() {
    rm -rf ../../build
    rm -rf ../../output
    rm -rf data
    bash ../../scripts/build.sh 36_w4a8_grouped_matmul_msd_full_loadA
}

function gen_data() {
    python3 gen_data.py $g ${kGroupSize} $m $n $k
    echo "Data gen finished"
}

function run_kernel {
    echo 'Case: m=' $m ' k=' $k ' n=' $n ' g=' $g ' kGroupSize=' ${kGroupSize}
    cd ../../output/bin/
    ./36_w4a8_grouped_matmul_msd_full_loadA $g ${kGroupSize} $m $n $k $device
}

build
gen_data
run_kernel
```

执行结果如下，说明精度比对成功。
```
Compare success.
```

## 约束限制
- n和k必须为偶数
- kGroupSize必须能被k整除
- l1Tile::K的值必须与kGroupSize相等