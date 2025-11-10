# EVG 在Fusion级的各组件简介


#### 1) `visitor_impl_base`
- `Arguments/Params`汇聚：统一参数管理
- `workspace计量`：对齐的workspace计算
- `can_implement`一致性校验：确保所有节点可执行

#### 2) `visitor_impl`
- 节点规范：统一的`get_callbacks`接口
- `Callbacks`生命周期钩子统一接口：`visit`/`begin_epilogue`/`end_epilogue`

- 由于需要进行自动流水管理，回调对象的visit函数带有VisitStage类型枚举参数，指示当前进行哪个阶段的指令操作。

```c++
enum class VisitStage : uint8_t {
    LOAD = 0,      // 执行所有load指令
    COMPUTE = 1,   // 执行所有compute指令
    STORE = 2,     // 执行所有store指令
};
```
根据自动流水管理的需求，每个节点的visit实现需要将AscendC指令包含在类似if (stage == VisitStage::COMPUTE) {}等条件语句中。否则无法保证流水的正确与双缓冲的形成。
一个节点可以包含在多个阶段中，比如broadcast节点可以同时包含LOAD阶段和COMPUTE阶段的指令。
PIPE内的同步指令需要在节点内进行，如AscendC::PipeBarrier<PIPE_V>();

#### 3) `tree_visitor`
- 参考模版形态，第一个为父节点
- 树形模板：父后子先遍历
- 按阶段传递子输出：收集子节点输出传给父节点

组合逻辑与 Arguments 顺序：
- 模板形态：`TreeVisitor<ParentOp, ChildOp1, ChildOp2, ...>`
- 参数顺序：`typename EVG::Arguments{ (ChildOp1::Arguments, ChildOp2::Arguments, ...), ParentOp::Arguments }`
- TreeVisitor 的 Arguments 书写顺序与模板声明相反——父节点的 Arguments 永远写在该层最后（先写所有子节点的 Arguments，再写父节点）。
- 示例：
```cpp
// (C + X) 写回 D（父最后写）
using EVG = TreeVisitor<
  VisitorAuxStore<half, LayoutC>,
  TreeVisitor<VisitorCompute<Add, half>, VisitorAccLoad<half>, VisitorAuxLoad<half, LayoutC>>
>;
typename EVG::Arguments args{
  { /* inner children then parent */ ArgsAccLoad{}, ArgsAuxLoad{deviceX, layoutD}, ArgsCompute{} },
  /* outer parent */ ArgsStore{deviceD, layoutD}
};
```
- 执行阶段：TreeVisitor 的回调以“先访问子节点、收集输出，再调用父节点”的顺序在每个阶段（LOAD/COMPUTE/STORE）分发。

#### 4) `topological_visitor`
- 参考模板形态，所以节点以拓扑序（依赖少的在前，依赖多的在后）平铺，最后一个节点为输出节点
- DAG拓扑复用：支持节点复用（如`(C+X)+(C+X)`）
- 节点缓存：按输出阶段缓存结果

组合逻辑与参数顺序：
- 模板形态：`TopologicalVisitor<EdgeTuple, Op0, Op1, ..., OpN>`，其中 `EdgeTuple` 指定每个索引的子依赖（最终根为索引 N）。
- 参数顺序：`typename EVG::Arguments{ Op0::Arguments, Op1::Arguments, ..., OpN::Arguments }`（严格按 `Ops...` 顺序）。
- 复用：同一算子类型可在 `Ops...` 中出现多次（如 Compute1/Compute2），其 Arguments 对应位置各自独立；回调执行时按 `EdgeTuple` 依赖关系决定访问序与缓存命中。
- 示例：
```cpp
using Edges = tla::tuple<
  tla::seq<>,        // 0: AccLoad
  tla::seq<>,        // 1: AuxLoad
  tla::seq<0, 1>,    // 2: Compute1 = C + X
  tla::seq<2, 2>,    // 3: Compute2 = (C+X) + (C+X)
  tla::seq<3>        // 4: Store
>;
using EVG = TopologicalVisitor<Edges,
  VisitorAccLoad<half>, VisitorAuxLoad<half, LayoutC>,
  VisitorCompute<Add, half>, VisitorCompute<Add, half>,
  VisitorAuxStore<half, LayoutC>
>;
typename EVG::Arguments args{ {}, {deviceX, layoutD}, {}, {}, {deviceD, layoutD} };
```
缓存语义：仅在各算子的输出阶段（非 STORE）缓存，避免覆盖带副作用的写回。

由于语义需要：每个节点需要在类内声明类似
- using ElementOutput = Element;

以便于TopologicalVisitor仅在各节点的输出阶段缓存节点的输出，而不用再次访问。
- static constexpr VisitStage OUTPUT_STAGE = VisitStage::COMPUTE;

以便于TopologicalVisitor申请对应输出类型的缓存引用对象。

#### 5) `visitor_compute`
- 须作为父节点使用
- 仅支持逐元素算子：如`Add`、`Sub`等
- 可接受任意个子节点，需要对应满足ComputeFn的参数要求，ComputeFn的operator()遵循目的在前，compute_length在中间，输入（可以是任意个）在后，例如：
```cpp
void operator()(
        AscendC::LocalTensor<T>& dst,
        uint32_t compute_length,
        AscendC::LocalTensor<T> const& src0,
        AscendC::LocalTensor<T> const& src1,
        Inputs const&... rest
)
```
- 类型匹配：确保输入类型一致

#### 6) `visitor_cast`

- 须作为父节点使用，仅支持一个子节点
- 类型转换：支持不同精度转换
- RoundMode：支持不同的舍入模式

#### 7) `visitor_acc_load`
- 可作为叶子节点独立存在
- 从GEMM工作区块按局部坐标Load
- 使用`gmSubblockC`和`layoutSubblockC`

#### 8) `visitor_aux_load`
- 可作为叶子节点独立存在
- 从用户GM按全局坐标Load
- 使用用户提供的`ptr_aux`和`layout`

#### 9) `visitor_aux_store`
- 须作为父节点使用，仅支持一个子节点，可继续作为子节点
- 写回用户GM（STORE阶段）
- 不直接配置原子；原子写回由`VisitorRowReduce`控制

#### 10) `visitor_row_broadcast`
- 可作为叶子节点独立存在，包含LOAD、COMPUTE双阶段，返回广播后的tile结果
- 1xN行向量广播到MxN tile
- 支持行复制

#### 11) `visitor_row_reduce`
- 须作为父节点使用，仅支持一个子节点，包含COMPUTE、STORE双阶段，可继续作为子节点，返回的是规约前的内容
- 按行规约到1xN并原子加到GM
- 当前仅支持tile间原子加出全局结果