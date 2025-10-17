#ifndef CATLASS_EPILOGUE_FUSION_OPERATIONS_HPP
#define CATLASS_EPILOGUE_FUSION_OPERATIONS_HPP

#include "catlass/catlass.hpp"

namespace Catlass::Epilogue::Fusion {

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

} // namespace Catlass::Epilogue::Fusion

#endif
