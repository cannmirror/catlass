# BasicMatmul Example Readme

## 功能说明

 - 算子功能：完成基础矩阵乘计算
 - 计算公式：
  $$
    C = A \times B
  $$
  其中$A$和$B$是输入矩阵，$C$是算子计算输出

## 参数说明

<table class="tg" style="undefined;table-layout: fixed; width: 500px"><colgroup>
<col style="width: 100px">
<col style="width: 200px">
<col style="width: 200px">
</colgroup>
<thead>
  <tr>
    <th class="tg-0pky">参数名</th>
    <th class="tg-0pky">描述</th>
    <th class="tg-0pky">约束</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-0pky"><code>m</code></td>
    <td class="tg-0pky">矩阵乘中左矩阵A的行</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>n</code></td>
    <td class="tg-0pky">矩阵乘中右矩阵B的列</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>k</code></td>
    <td class="tg-0pky">矩阵乘中左矩阵A的列（也即右矩阵的行数）</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>deviceId</code></td>
    <td class="tg-0pky">使用的NPU卡ID（默认0）</td>
    <td class="tg-0pky">在设备的NPU有效范围内</td>
  </tr>  

</tbody>
</table>

相应地，原型计算有如下限制：

|名称/Name|类型/Class|数据类型/Dtype|维度/Dims|格式/Format|描述/Description|
|---|---|---|---|---|---|
|matA|inTensor|int8\|fp16\|bf16\|fp32|[m, k]|ND\|NZ|左矩阵，支持转置|
|matB|inTensor|int8\|fp16\|bf16\|fp32|[n, k]|ND\|NZ|右矩阵，支持转置|
|matC|outTensor|fp16\|bf16|[m, n]|ND|输出矩阵，非转置|

## 约束说明

无

## 代码组织
```
├── 00_basic_matmul
│   ├── CMakeLists.txt   # CMake编译文件
│   ├── README.md
│   └── basic_matmul.cpp # 主文件
```

## 使用示例
1. 编译样例代码，并编译生成相应的算子可执行文件。
```
bash scripts/build.sh 00_basic_matmul
```

2. 切换到可执行文件的编译目录`output/bin`下，执行算子样例程序。测试样例数据随机生成，尺寸从命令行输入。
```
cd output/bin
./00_basic_matmul 256 512 1024 0
```
• 256：矩阵m轴

• 512：n轴

• 1024：k轴

• 0：Device ID，可选，默认为0


执行结果如下，说明样例执行成功。
```
Compare success.
```