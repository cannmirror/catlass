# BasicMatmul Example Readme

## 功能说明
> *资料书写*
> - 介绍算子功能，使用场景及注意事项
> - 罗列计算公式或计算流程；要求能用数学表达式或流程图阐述算子功能

 - 算子功能：完成基础矩阵乘计算
 - 计算公式：
  $$
    C = A \times B
  $$
  其中$A$和$B$是输入矩阵，$C$是算子计算输出

## 参数说明

> *资料书写要求：*
> 
>**本章定位：** 并非解释算子，而是解释上文算子公式或流程图里的变量含义，变量与参数名需一一对应，目标就是解释清楚算子的含义和实现的功能。
>- 参数说明表格呈现，列出每个参数的名称以及描述。
>><span style="color: red;">注意：不要在表格里区分芯片描述，差异化描述在表格外，以“产品1、产品2：xx描述”形式组织。</span>
>- 格式要求：表格中无内容用-填充，"描述"列加标点符号，“参数名”列内容不可换行显示，其它列宽见下述要求。
>- 表格自动生成md工具：https://www.tablesgenerator.com/html_tables。
>- 缩进要求：表格不缩进，与“参数说明”文字对齐，表格下文字不缩进。

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
    <td class="tg-0pky">m</td>
    <td class="tg-0pky">矩阵乘中左矩阵A的行</td>
    <td class="tg-0pky">在<code>uint32_t</code>范围内</td>
  </tr>
  <tr>
    <td class="tg-0pky">n</td>
    <td class="tg-0pky">矩阵乘中右矩阵B的列</td>
    <td class="tg-0pky">在<code>uint32_t</code>范围内</td>
  </tr>
  <tr>
    <td class="tg-0pky">k</td>
    <td class="tg-0pky">矩阵乘中左矩阵A的列（也即右矩阵的行数）</td>
    <td class="tg-0pky">在<code>uint32_t</code>范围内</td>
  </tr>
  <tr>
    <td class="tg-0pky">deviceId</td>
    <td class="tg-0pky">使用的NPU卡ID（默认0）</td>
    <td class="tg-0pky">在设备的NPU有效范围内</td>
  </tr>  

</tbody>
</table>

## 约束说明

> *资料书写要求*  
> 通用性约束（算子原生语义已有、竞品同样存在的约束）无需重复说明，避免赘述使用说明中的内容。应从使用场景、硬件/软件资源、系统/网络性能或精度影响等角度进行描述。若无相关约束，应填写“无”，章节不允许为空。内容以无序列表形式呈现，条目末尾需加标点符号。  
> 写作清单<span style="color: red;">（以下为算子示例，需根据实际情况调整）</span>：

> - 确定性实现 / 非确定性实现；
> - 输入形状限制：如存在超出原生语义的维度限制，需在此说明；
> - 输入值域限制：输入数据若在特定范围外存在精度或性能问题，需说明有效值域；
> - 输入属性限制：输入属性若超出某值域会被拦截报错，需说明限制条件；
> - 输入数据类型限制：如某些数据类型在特定模式下性能显著下降，需特别标注；
> - 其他限制。

无

## 代码组织
```
├── 00_basic_matmul
│   ├── CMakeLists.txt   # CMake编译文件
│   ├── README.md
│   └── basic_matmul.cpp # 主文件
```
## 使用示例
- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/quickstart.md#算子编译)
- 执行算子
```
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