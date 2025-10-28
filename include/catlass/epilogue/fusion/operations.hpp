#ifndef CATLASS_EPILOGUE_FUSION_OPERATIONS_HPP
#define CATLASS_EPILOGUE_FUSION_OPERATIONS_HPP

#include "catlass/catlass.hpp"

namespace Catlass::Epilogue::Fusion {
// 指数一元算子
template <typename T>
struct ExpOp {
    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<T>& dst,
                    AscendC::LocalTensor<T> const& src,
                    uint32_t compute_length) const {
        AscendC::Exp(dst, src, compute_length);
    }
};

// 减法二元算子 dst = src0 - src1
template <typename T>
struct SubOp {
    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<T>& dst,
                    AscendC::LocalTensor<T> const& src0,
                    AscendC::LocalTensor<T> const& src1,
                    uint32_t compute_length) const {
        AscendC::Sub(dst, src0, src1, compute_length);
    }
};

// 除法二元算子 dst = src0 / src1
template <typename T>
struct DivOp {
    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<T>& dst,
                    AscendC::LocalTensor<T> const& src0,
                    AscendC::LocalTensor<T> const& src1,
                    uint32_t compute_length) const {
        AscendC::Div(dst, src0, src1, compute_length);
    }
};


// 类型转换操作符（3参数版本）
template <
    typename T,
    typename S,
    AscendC::RoundMode RoundMode = AscendC::RoundMode::CAST_NONE
>
struct NumericArrayConverter {
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<T>& dst,
        AscendC::LocalTensor<S> const& src,
        uint32_t compute_length
    ) const {
        if constexpr (std::is_same_v<T, S>) {
            AscendC::DataCopy(dst, src, compute_length);
        } else {
            AscendC::Cast(dst, src, RoundMode, compute_length);
        }
    }
};

// 加法操作符
template <typename T>
struct Plus {
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<T>& dst,
        AscendC::LocalTensor<T> const& src0,
        AscendC::LocalTensor<T> const& src1,
        uint32_t compute_length
    ) const {
        AscendC::Add(dst, src0, src1, compute_length);
    }
};

// 最大值操作符
template <typename T>
struct Maximum {
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<T>& dst,
        AscendC::LocalTensor<T> const& src0,
        AscendC::LocalTensor<T> const& src1,
        uint32_t compute_length
    ) const {
        AscendC::Max(dst, src0, src1, compute_length);
    }
};

// 最小值操作符
template <typename T>
struct Minimum {
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<T>& dst,
        AscendC::LocalTensor<T> const& src0,
        AscendC::LocalTensor<T> const& src1,
        uint32_t compute_length
    ) const {
        AscendC::Min(dst, src0, src1, compute_length);
    }
};

// 规约策略枚举
enum class ReduceStrategy {
    ATOMIC_REDUCE,      // 原子加：每个 block 直接原子累加到 GM 输出
    WORKSPACE_REDUCE // Workspace 规约：先写 workspace，最后统一规约
};

// 规约操作函数对象基类
template <template<class> class ReduceFn, typename Element>
struct ReduceOp {
    // 行规约：二元操作
    CATLASS_DEVICE static void row_reduce(
        AscendC::LocalTensor<Element>& dst,
        AscendC::LocalTensor<Element> const& src0,
        AscendC::LocalTensor<Element> const& src1,
        uint32_t length) {
        static_assert(sizeof(ReduceFn<Element>) == 0, "Unsupported reduce operation");
    }
    
    // 列/标量规约：硬件加速
    CATLASS_DEVICE static void vector_reduce(
        AscendC::LocalTensor<Element>& dst,
        AscendC::LocalTensor<Element> const& src,
        AscendC::LocalTensor<Element>& work,
        uint32_t length) {
        static_assert(sizeof(ReduceFn<Element>) == 0, "Unsupported reduce operation");
    }
};

// Plus 特化
template <typename Element>
struct ReduceOp<Plus, Element> {
    CATLASS_DEVICE static void row_reduce(
        AscendC::LocalTensor<Element>& dst,
        AscendC::LocalTensor<Element> const& src0,
        AscendC::LocalTensor<Element> const& src1,
        uint32_t length) {
        AscendC::Add(dst, src0, src1, length);
    }
    
    CATLASS_DEVICE static void vector_reduce(
        AscendC::LocalTensor<Element>& dst,
        AscendC::LocalTensor<Element> const& src,
        AscendC::LocalTensor<Element>& work,
        uint32_t length) {
        AscendC::ReduceSum(dst, src, work, length);
    }
};

// Maximum 特化
template <typename Element>
struct ReduceOp<Maximum, Element> {
    CATLASS_DEVICE static void row_reduce(
        AscendC::LocalTensor<Element>& dst,
        AscendC::LocalTensor<Element> const& src0,
        AscendC::LocalTensor<Element> const& src1,
        uint32_t length) {
        AscendC::Max(dst, src0, src1, length);
    }
    
    CATLASS_DEVICE static void vector_reduce(
        AscendC::LocalTensor<Element>& dst,
        AscendC::LocalTensor<Element> const& src,
        AscendC::LocalTensor<Element>& work,
        uint32_t length) {
        AscendC::ReduceMax(dst, src, work, length);
    }
};

// Minimum 特化
template <typename Element>
struct ReduceOp<Minimum, Element> {
    CATLASS_DEVICE static void row_reduce(
        AscendC::LocalTensor<Element>& dst,
        AscendC::LocalTensor<Element> const& src0,
        AscendC::LocalTensor<Element> const& src1,
        uint32_t length) {
        AscendC::Min(dst, src0, src1, length);
    }
    
    CATLASS_DEVICE static void vector_reduce(
        AscendC::LocalTensor<Element>& dst,
        AscendC::LocalTensor<Element> const& src,
        AscendC::LocalTensor<Element>& work,
        uint32_t length) {
        AscendC::ReduceMin(dst, src, work, length);
    }
};

// 规约计算策略 traits 基类
template <template<class> class ReduceFn, typename Element, ReduceStrategy Strategy>
struct ReduceComputeStrategy {
    static constexpr ReduceStrategy strategy = Strategy;
    
    // 默认实现：编译期错误提示
    template <typename... Args>
    CATLASS_DEVICE static void compute(Args&&...) {
        static_assert(sizeof(ReduceFn<Element>) == 0, "Unsupported reduce operation for this strategy");
    }
    
    template <typename... Args>
    CATLASS_DEVICE static void store(Args&&...) {
        static_assert(sizeof(ReduceFn<Element>) == 0, "Unsupported reduce operation for this strategy");
    }
};

// 原子操作设置traits
template <template<class> class ReduceFn, typename T>
struct AtomicSetter {
    CATLASS_DEVICE
    static void set() { /* default: no atomic */ }
    CATLASS_DEVICE
    static void clear() { AscendC::SetAtomicNone(); }
};

// Plus特化
template <typename T>
struct AtomicSetter<Plus, T> {
    CATLASS_DEVICE
    static void set() { AscendC::SetAtomicAdd<T>(); }
    CATLASS_DEVICE
    static void clear() { AscendC::SetAtomicNone(); }
};

// Maximum特化
template <typename T>
struct AtomicSetter<Maximum, T> {
    CATLASS_DEVICE
    static void set() { AscendC::SetAtomicMax<T>(); }
    CATLASS_DEVICE
    static void clear() { AscendC::SetAtomicNone(); }
};

// Minimum特化
template <typename T>
struct AtomicSetter<Minimum, T> {
    CATLASS_DEVICE
    static void set() { AscendC::SetAtomicMin<T>(); }
    CATLASS_DEVICE
    static void clear() { AscendC::SetAtomicNone(); }
};

// RowReduceCompute 策略 traits - 统一实现
template <template<class> class ReduceFn, typename Element, ReduceStrategy Strategy>
struct RowReduceCompute {
    static constexpr ReduceStrategy strategy = Strategy;
    
    CATLASS_DEVICE static void compute(
        AscendC::LocalTensor<Element>& ubReduce,
        AscendC::LocalTensor<Element> const& input,
        Element const& identity,
        uint32_t actualRows,
        uint32_t actualCols,
        uint32_t alignedCols)
    {
        // 初始化 reduce buffer 为 identity
        AscendC::Duplicate(ubReduce, identity, alignedCols);
        if (actualRows > 0) {
            // 第一行：直接复制
            AscendC::DataCopy(ubReduce, input, actualCols);
        }
        
        // 后续行：规约操作
        for (uint32_t r = 1; r < actualRows; ++r) {
            AscendC::PipeBarrier<PIPE_V>();
            // 使用函数对象，无需特化三次
            ReduceOp<ReduceFn, Element>::row_reduce(
                ubReduce, ubReduce, input[r * alignedCols], actualCols);
        }
    }
    
    CATLASS_DEVICE static void store(
        AscendC::GlobalTensor<Element>& gmOut,
        AscendC::LocalTensor<Element> const& ubReduce,
        MatrixCoord const& globalTileOffset,
        MatrixCoord const& actualTileShape,
        layout::RowMajor const& layout)
    {
        auto gmTile = gmOut[layout.GetOffset(MatrixCoord{0, globalTileOffset.column()})];
        auto layoutUbRowOut = layout::RowMajor::MakeLayoutInUb<Element>(MatrixCoord{1u, actualTileShape.column()});
        using CopyUb2GmT = Epilogue::Tile::CopyUb2Gm<Arch::AtlasA2, Gemm::GemmType<Element, layout::RowMajor>>;
        CopyUb2GmT copyUb2Gm{};
        auto layoutDst = layout.GetTileLayout(MatrixCoord{1u, actualTileShape.column()});
        
        AtomicSetter<ReduceFn, Element>::set();
        copyUb2Gm(gmTile, ubReduce, layoutDst, layoutUbRowOut);
        AtomicSetter<ReduceFn, Element>::clear();
    }
};

// ColReduceCompute 策略 traits - 统一实现
template <template<class> class ReduceFn, typename Element, ReduceStrategy Strategy>
struct ColReduceCompute {
    static constexpr ReduceStrategy strategy = Strategy;
    
    CATLASS_DEVICE static void compute(
        AscendC::LocalTensor<Element>& ubRowReduce,
        AscendC::LocalTensor<Element>& ubWork,
        AscendC::LocalTensor<Element> const& input,
        uint32_t actualRows,
        uint32_t actualCols,
        uint32_t alignedCols)
    {
        // Reduce each row of the tile over columns to a scalar
        for (uint32_t r = 0; r < actualRows; ++r) {
            // 使用函数对象，无需特化三次
            auto dst = ubRowReduce[r*alignedCols];
            ReduceOp<ReduceFn, Element>::vector_reduce(
                dst, input[r * alignedCols], ubWork, actualCols);
        }
    }
    
    CATLASS_DEVICE static void store(
        AscendC::GlobalTensor<Element>& gmOut,
        AscendC::LocalTensor<Element> const& ubRowReduce,
        MatrixCoord const& globalTileOffset,
        MatrixCoord const& actualTileShape,
        layout::RowMajor const& layout)
    {
        auto gmTile = gmOut[layout.GetOffset(MatrixCoord{globalTileOffset.row(), 0})];
        auto layoutUbCol = layout::RowMajor{actualTileShape.row(), 1, actualTileShape.column()};
        using CopyUb2GmT = Epilogue::Tile::CopyUb2Gm<Arch::AtlasA2, Gemm::GemmType<Element, layout::RowMajor>>;
        CopyUb2GmT copyUb2Gm{};
        auto layoutDst = layout.GetTileLayout(MatrixCoord{actualTileShape.row(), 1u});
        
        AtomicSetter<ReduceFn, Element>::set();
        copyUb2Gm(gmTile, ubRowReduce, layoutDst, layoutUbCol);
        AtomicSetter<ReduceFn, Element>::clear();
    }
};

// ScalarReduceCompute 策略 traits - 统一实现
template <template<class> class ReduceFn, typename Element, ReduceStrategy Strategy>
struct ScalarReduceCompute {
    static constexpr ReduceStrategy strategy = Strategy;
    
    CATLASS_DEVICE static void compute(
        AscendC::LocalTensor<Element>& ubScalar,
        AscendC::LocalTensor<Element>& ubWork,
        AscendC::LocalTensor<Element> const& input,
        Element const& identity,
        uint32_t actualRows,
        uint32_t actualCols)
    {
        AscendC::Duplicate(ubScalar, identity, actualRows * actualCols);
        // 使用函数对象，无需特化三次
        ReduceOp<ReduceFn, Element>::vector_reduce(ubScalar, input, ubWork, actualRows * actualCols);
    }
    
    CATLASS_DEVICE static void store(
        AscendC::GlobalTensor<Element>& gmOut,
        AscendC::LocalTensor<Element> const& ubScalar,
        layout::RowMajor const& layout)
    {
        auto gmTile = gmOut[layout.GetOffset(MatrixCoord{0u, 0u})];
        using CopyUb2GmT = Epilogue::Tile::CopyUb2Gm<Arch::AtlasA2, Gemm::GemmType<Element, layout::RowMajor>>;
        CopyUb2GmT copyUb2Gm{};
        auto layoutUbScalar = layout::RowMajor::MakeLayoutInUb<Element>(MatrixCoord{1u, 1u});
        auto layoutDst = layout.GetTileLayout(MatrixCoord{1u, 1u});
        
        AtomicSetter<ReduceFn, Element>::set();
        copyUb2Gm(gmTile, ubScalar, layoutDst, layoutUbScalar);
        AtomicSetter<ReduceFn, Element>::clear();
    }
};

} // namespace Catlass::Epilogue::Fusion

#endif
