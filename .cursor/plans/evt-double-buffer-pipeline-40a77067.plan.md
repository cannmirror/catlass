<!-- 40a77067-5eca-4438-ba4d-a61ab71f0ff5 25901eef-2b0d-4a83-8ea6-a4dcd31a5c5b -->
# EVT 双缓冲流水线实现方案

## 核心设计思想

参考 `block_epilogue_per_token_dequant.hpp` 的多缓冲实现，将整个EVT树视为**Load→Compute→Store三个阶段的流水**，采用经典的双缓冲同步策略：

```
Ping Buffer: Load[0] → Compute[0] → Store[0]
Pong Buffer:            Load[1] → Compute[1] → Store[1]
```

**关键优化**：

- 使用 **4个event_id** 实现双缓冲同步（每个buffer用2个：MTE2_V和V_MTE3）
- 在 `BlockEpilogue` 层面分三次调用 `visit`，通过参数控制节点执行阶段
- 同步逻辑从节点内部提升到 `BlockEpilogue`，实现阶段间流水并行

## 实现步骤

### 1. 定义访问阶段枚举类型

**文件**: `include/catlass/epilogue/fusion/visitor_impl_base.hpp`

在命名空间 `Catlass::Epilogue::Fusion` 中添加：

```cpp
enum class VisitStage : uint8_t {
    LOAD = 0,      // 执行所有load节点
    COMPUTE = 1,   // 执行所有compute节点
    STORE = 2,     // 执行所有store节点
    ALL = 3        // 单缓冲模式（默认，兼容旧代码）
};
```

### 2. 修改所有visitor节点的visit方法签名

为所有节点的 `Callbacks::visit` 方法添加 `VisitStage stage` 参数（默认值为 `ALL`）。

#### 2.1 修改 VisitorAuxLoad

**文件**: `include/catlass/epilogue/fusion/visitor_aux_load.hpp`

**修改点**:

- 第75-81行：在 `visit` 方法签名中添加 `VisitStage stage = VisitStage::ALL` 参数
- 第82-111行：整个方法体用条件包裹：`if (stage == VisitStage::LOAD || stage == VisitStage::ALL)`
- **移除内部同步**：删除 82-83行、107-108行的 `SetFlag/WaitFlag`（同步逻辑移到BlockEpilogue）

#### 2.2 修改 VisitorCompute

**文件**: `include/catlass/epilogue/fusion/visitor_compute.hpp`

**修改点**:

- 第113行：添加 `VisitStage stage = VisitStage::ALL` 参数
- 第120-141行：整个方法体用条件包裹：`if (stage == VisitStage::COMPUTE || stage == VisitStage::ALL)`
- **移除内部同步**：删除 120-121行的 `SetFlag/WaitFlag`

#### 2.3 修改 VisitorAuxStore

**文件**: `include/catlass/epilogue/fusion/visitor_aux_store.hpp`

**修改点**:

- 第73行：添加 `VisitStage stage = VisitStage::ALL` 参数
- 第79-107行：整个方法体用条件包裹：`if (stage == VisitStage::STORE || stage == VisitStage::ALL)`
- **移除内部同步**：删除 89-90行的 `SetFlag/WaitFlag`

### 3. 修改TreeVisitor传递stage参数

**文件**: `include/catlass/epilogue/fusion/tree_visitor.hpp`

**修改点**:

- 第21-34行 `collect_child_outputs`：添加 `VisitStage stage` 参数，传递给子节点visit
- 第37-51行 `call_parent_with_outputs`：添加 `VisitStage stage` 参数，传递给父节点visit
- 第55-77行 `visit`：添加 `VisitStage stage = VisitStage::ALL` 参数，传递给辅助函数

### 4. 实现双缓冲流水的BlockEpilogue

**文件**: `include/catlass/epilogue/block/block_epilogue_visitor.hpp`

参考 `block_epilogue_per_token_dequant.hpp:127-181` 和 `230-298` 的实现模式。

#### 4.1 修改构造函数（第48-57行）

初始化双缓冲的event_id：

```cpp
CATLASS_DEVICE
BlockEpilogue(Arch::Resource<ArchTag>& resource, Params const& params)
    : params(params), fusion_callbacks(params.fusion_params)
{
    // 为两个buffer分配独立的event_id
    eventMTE2V[0] = 0;
    eventMTE2V[1] = 1;
    eventVMTE3[0] = 2;
    eventVMTE3[1] = 3;
    
    // 初始状态：允许搬入和搬出
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventMTE2V[0]);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventMTE2V[1]);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventVMTE3[0]);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventVMTE3[1]);
}
```

#### 4.2 修改析构函数（第54-57行）

等待所有流水完成：

```cpp
CATLASS_DEVICE
~BlockEpilogue()
{
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventMTE2V[0]);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventMTE2V[1]);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventVMTE3[0]);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventVMTE3[1]);
}
```

#### 4.3 修改operator()实现三阶段流水（第60-174行）

核心逻辑改造为：

```cpp
// 遍历所有 tile，实现双缓冲流水
uint32_t tileIdx = 0;
uint32_t ubListId = 0;  // 0或1，交替使用

for (uint32_t r = 0; r < rows; ) {
    // ... 计算tileShape, globalTileOffset等 ...
    
    // === Load阶段：等待上一次compute完成 ===
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventMTE2V[ubListId]);
    
    // 调用EVT树load阶段（MTE2流水）
    auto& cbs = ((ubListId & 1) ? callbacks1 : callbacks0);
    cbs.visit(globalTileOffset, localTileOffset, tileShape, calCount, 
              VisitStage::LOAD);
    
    // Load完成，通知compute可以开始
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventMTE2V[ubListId]);
    
    // === Compute阶段：等待load完成 & store空闲 ===
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventMTE2V[ubListId]);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventVMTE3[ubListId]);
    
    // 调用EVT树compute阶段（V流水）
    cbs.visit(globalTileOffset, localTileOffset, tileShape, calCount,
              VisitStage::COMPUTE);
    
    // Compute完成，通知load可以覆盖 & store可以搬出
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventMTE2V[ubListId]);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventVMTE3[ubListId]);
    
    // === Store阶段：等待compute完成 ===
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventVMTE3[ubListId]);
    
    // 调用EVT树store阶段（MTE3流水）
    cbs.visit(globalTileOffset, localTileOffset, tileShape, calCount,
              VisitStage::STORE);
    
    // Store完成，通知compute可以写入
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventVMTE3[ubListId]);
    
    // Buffer轮转
    ubListId = 1 - ubListId;
    ++tileIdx;
    r += tileRows;
}
```

#### 4.4 添加私有成员变量（第176-179行）

```cpp
private:
    Params params;
    FusionCallbacks fusion_callbacks;
    int32_t eventMTE2V[2];  // MTE2→V同步事件（两个buffer）
    int32_t eventVMTE3[2];  // V→MTE3同步事件（两个buffer）
```

### 5. 更新EmptyCallbacks支持stage参数

**文件**: `include/catlass/epilogue/fusion/visitor_impl.hpp`

第11-16行的 `EmptyCallbacks` 无需修改（空实现不受影响）。

## 关键设计要点

### 同步策略对照表

| 阶段 | Wait事件 | 操作 | Set事件 | 语义 |

|------|---------|------|---------|------|

| Load | `V_MTE2` | MTE2搬入 | `MTE2_V` | "上次compute完成，我可以搬入；搬入完成，通知compute" |

| Compute | `MTE2_V` + `MTE3_V` | V计算 | `V_MTE2` + `V_MTE3` | "load完成且store空闲，我可以计算；计算完成，允许覆盖和搬出" |

| Store | `V_MTE3` | MTE3搬出 | `MTE3_V` | "compute完成，我可以搬出；搬出完成，buffer空闲" |

### Event ID 分配方案

- **eventMTE2V[0]** = EVENT_ID0: ping buffer的load→compute同步
- **eventMTE2V[1]** = EVENT_ID1: pong buffer的load→compute同步
- **eventVMTE3[0]** = EVENT_ID2: ping buffer的compute→store同步
- **eventVMTE3[1]** = EVENT_ID3: pong buffer的compute→store同步

总计使用4个event_id，远低于16个限制。

### 兼容性保证

- 默认 `stage = VisitStage::ALL` 时，保持单缓冲行为（所有节点同步执行）
- 现有代码无需修改即可继续工作
- 双缓冲模式仅在BlockEpilogue层面显式调用三阶段visit时生效

## 测试验证

使用 `examples/32_matmul_add_evt/matmul_add_evt.cpp` 验证：

1. 编译通过
2. 功能正确性（与golden结果对比）
3. 性能提升（通过NPU profiling观察流水并行度）

### To-dos

- [ ] 在 visitor_impl_base.hpp 中添加 VisitStage 枚举类型定义
- [ ] 修改 VisitorAuxLoad 的 visit 方法：添加 stage 参数，移除内部同步，添加阶段条件判断
- [ ] 修改 VisitorCompute 的 visit 方法：添加 stage 参数，移除内部同步，添加阶段条件判断
- [ ] 修改 VisitorAuxStore 的 visit 方法：添加 stage 参数，移除内部同步，添加阶段条件判断
- [ ] 修改 TreeVisitor 的 Callbacks::visit 和辅助方法，传递 stage 参数到子节点和父节点
- [ ] 重构 BlockEpilogue::operator()：实现三阶段visit调用和双缓冲同步逻辑，参考 per_token_dequant 实现
- [ ] 修改 BlockEpilogue 构造函数、析构函数和私有成员，添加双缓冲event_id管理
- [ ] 编译并运行 matmul_add_evt 示例，验证双缓冲流水的功能正确性和性能