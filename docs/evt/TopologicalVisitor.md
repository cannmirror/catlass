## TopologicalVisitor 设计说明

### 背景与目标

TopologicalVisitor 提供了一种“按拓扑依赖关系”访问并调度后处理（Epilogue）算子回调的机制。与传统的编译期树形组合（TreeVisitor）不同，TopologicalVisitor 使用一个边集描述（Edges）来表达各节点间的依赖关系，并按照拓扑顺序递归调用各子节点，最终将子节点输出按声明顺序传递给父节点完成计算及存储。

典型用途：在 GEMM 之后执行“加载累加结果 + 加载辅助张量 + 逐元素计算 + 写回”的算子序列，且希望用统一的机制表达依赖与执行顺序。

### 模板参数与类型

- EdgeTuple：依赖描述元组，类型为 `tla::tuple<tla::seq<child_idx...>, ...>`，长度等于算子个数 R；第 i 个元素是 `tla::seq<...>`，其中的每个整型均是该节点的子节点索引；索引从 0 开始。
- Ops...：按顺序列出参与拓扑图的算子类型（每个算子均为 Visitor 节点类型，如 `VisitorAccLoad`、`VisitorAuxLoad`、`VisitorCompute`、`VisitorAuxStore`、`VisitorCast` 等）。约定最后一个算子（索引 R-1）是“根”，也就是最终输出算子（例如 Store）。

TopologicalVisitor 继承 `VisitorImpl<Ops...>`，重用参数打包、可实现性检查、工作区（workspace）大小计算以及回调组装等基础逻辑。

### 内部结构与子成员逻辑

TopologicalVisitor 内部定义了一个嵌套结构体 `Callbacks`，该结构体继承自 `VisitorImpl<Ops...>::Callbacks<...>` 的实现实例（以下简称 `CallbacksImpl`），并在此基础上扩展了基于拓扑的访问函数族：

- callbacks_tuple（继承自 CallbacksImpl）：
  - 各算子的回调实例以元组形式存放，顺序与模板参数 Ops... 一致。

- visit(...)：统一入口
  - 签名：`visit(tileOffset, localTileOffset, tileShape, calCount, stage, args...)`。
  - 逻辑：以 R=sizeof...(Ops) 计算根索引 R-1，取出 `RootEdges = get<R-1>(EdgeTuple{})`，调用 `visit_node<R-1>(..., RootEdges, args...)` 发起整棵（或整张 DAG 的根）访问。

- visit_node<I, Seq>(...)
  - 作用：访问索引为 I 的算子节点。首先按 `Seq`（一个 `tla::seq<child_idx...>`）递归访问所有子节点，收集其输出，再以这些输出作为尾部参数调用第 I 个算子的回调 `visit(...)`。
  - 关键步骤：
    1) `collect_children<I>(..., Seq, args...)` 将子节点输出收集到一个 `tla::tuple<...>` 中；
    2) `call_current<I>(..., child_outputs)` 将 `child_outputs` 展开为参数包并调用 `tla::get<I>(callbacks_tuple).visit(...)`。

- collect_children<I, ChildIs...>(..., tla::seq<ChildIs...>, args...)
  - 作用：对 `Seq` 中的每个 `ChildIs` 递归调用 `visit_node<ChildIs>(..., ChildEdges, args...)`，其中 `ChildEdges = decltype(tla::get<ChildIs>(EdgeTuple{}))`（该子节点的子边集）。
  - 返回值：`tla::tuple<Child0Output, Child1Output, ...>`，输出顺序与 `Seq` 中的子索引顺序一致。

- call_current<I>(..., ChildOutputs)
  - 作用：将 `ChildOutputs`（一个 `tla::tuple`）展开为参数包，调用第 I 个回调：
    - 内部通过 `call_current_expand<I>(..., child_outputs, tla::make_seq<Num>{})` 生成索引序列 `0..Num-1`；
    - 再以 `tla::get<Js>(child_outputs)...` 展开为可变参数，调用 `tla::get<I>(callbacks_tuple).visit(...)`。
  - 特性：当 `Num=0`（无子节点）时，参数包为空；这对诸如 `VisitorAccLoad`、`VisitorAuxLoad` 之类无子输入的算子是必需的，能够避免编译器匹配错误。

- get_callbacks(...)
  - 作用：向下调用基类 `VisitorImpl<Ops...>::get_callbacks(...)` 生成 `CallbacksImpl`，然后包装为本类型的 `Callbacks` 以获得拓扑访问功能。UB 申请、布局信息、资源句柄均由各子算子的回调工厂负责，TopologicalVisitor 不额外分配资源。

### 阶段（VisitStage）与数据流

- TopologicalVisitor 将 `VisitStage`（LOAD / COMPUTE / STORE / ALL）透明地传递给子节点回调；是否在当前阶段执行、执行什么操作由各子算子回调自行判断（例如 `VisitorAccLoad` 仅在 LOAD 阶段从 GM 读入；`VisitorCompute` 仅在 COMPUTE 阶段计算；`VisitorAuxStore` 仅在 STORE 阶段写回 GM）。
- 由于 `BlockEpilogue` 在外层以双缓冲流水调度（MTE2/V/MTE3 分阶段同步），TopologicalVisitor 仅负责“算子间依赖顺序与参数传递”，不会改变既有的流水与同步语义。

### 类型与约束

- 输入/输出类型匹配：
  - `VisitorCompute<Fn, T>` 要求其所有子输入均为 `T`；如不满足，请显式插入 `VisitorCast<T, S>` 进行类型转换。
  - `VisitorAuxStore<T, ...>` 要求输入为 `T`；否则同样需要 `VisitorCast`。
- 拓扑约束：
  - EdgeTuple 必须无环；最后一个算子（索引 R-1）是根。
  - 若一个子节点被多个父节点复用，当前实现会对该子节点执行多次递归访问（不做结果缓存）；若出现性能瓶颈，可在后续版本扩展“结果缓存”机制。

### 与 TreeVisitor 的差异

- TreeVisitor 通过模板嵌套表达固定的树形结构，父节点类型出现在模板参数最后；
- TopologicalVisitor 通过 EdgeTuple 显式描述依赖关系，不再要求编译期的树形嵌套结构，在表达更复杂的（无环）图结构时更灵活；
- 二者都重用同一套回调工厂与阶段语义，易于互换与对比验证。

### 使用示例（与示例 34 一致）

以 `D = C + X` 为例，定义 4 个节点：`AccLoad`、`AuxLoad`、`Compute(Plus)`、`Store`，并以 `Store` 为根。

```cpp
using Edges = tla::tuple<
    tla::seq<>,      // 0: AccLoad 无子节点
    tla::seq<>,      // 1: AuxLoad 无子节点
    tla::seq<0, 1>,  // 2: Compute 依赖 AccLoad 与 AuxLoad
    tla::seq<2>      // 3: Store 依赖 Compute
>;

using EVT = Epilogue::Fusion::TopologicalVisitor<
    Edges,
    Epilogue::Fusion::VisitorAccLoad<half>,
    Epilogue::Fusion::VisitorAuxLoad<half, layout::RowMajor>,
    Epilogue::Fusion::VisitorCompute<Epilogue::Fusion::Plus, half>,
    Epilogue::Fusion::VisitorAuxStore<half, layout::RowMajor>
>;

typename EVT::Arguments evt_args{
    Epilogue::Fusion::VisitorAccLoad<half>::Arguments{},
    Epilogue::Fusion::VisitorAuxLoad<half, layout::RowMajor>::Arguments{deviceX, layoutD},
    Epilogue::Fusion::VisitorCompute<Epilogue::Fusion::Plus, half>::Arguments{},
    Epilogue::Fusion::VisitorAuxStore<half, layout::RowMajor>::Arguments{deviceD, layoutD}
};
```

### 实现注意事项

- `Evt::Arguments` 的参数顺序必须与 `Ops...` 的顺序完全一致。
- `Edges` 中的子节点索引必须落在 `[0, R-1]` 范围内，并且不得形成环。
- 若需要跨类型组合（例如 `float + half`），请在相应路径插入 `VisitorCast` 做类型转换，TopologicalVisitor 本身不做统一精度缓冲。

### 维护与文件位置

- 头文件：`include/catlass/epilogue/fusion/topological_visitor.hpp`
- 汇总头引入：`include/catlass/epilogue/fusion/fusion.hpp`
- 示例：`examples/34_matmul_add_topo/matmul_add_topo.cpp`

### 变更记录

- v1.0 首次引入：
  - 基于 Edges 的拓扑递归访问；
  - 支持无子节点（空参数包）路径；
  - 与 BlockEpilogue 的三阶段流水无缝对接；
  - 与 TreeVisitor 行为等价（在树结构场景），支持更灵活的无环依赖表达。


