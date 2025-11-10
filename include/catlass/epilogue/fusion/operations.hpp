#ifndef CATLASS_EPILOGUE_FUSION_OPERATIONS_HPP
#define CATLASS_EPILOGUE_FUSION_OPERATIONS_HPP

#include "catlass/catlass.hpp"

namespace Catlass::Epilogue::Fusion {

// 一元
template <typename T>
struct Exp {
    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<T>& dst,
                    uint32_t compute_length,
                    AscendC::LocalTensor<T> const& src
    ) const {
        AscendC::Exp(dst, src, compute_length);
    }
};


template <typename T>
struct Relu {
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<T>& dst,
        uint32_t compute_length,
        AscendC::LocalTensor<T> const& src
    ) const {
        AscendC::Relu(dst, src, compute_length);
    }
};


template <
    typename T,
    typename S,
    AscendC::RoundMode RoundMode = AscendC::RoundMode::CAST_NONE
>
struct Cast {
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<T>& dst,
        uint32_t compute_length,
        AscendC::LocalTensor<S> const& src
    ) const {
        static_assert(!std::is_same_v<T, S>, "Cast: input type mismatch");
        AscendC::Cast(dst, src, RoundMode, compute_length);
    }
};

// 二元
template <typename T>
struct Mul {
    template <typename... Inputs>
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<T>& dst,
        uint32_t compute_length,
        AscendC::LocalTensor<T> const& src0,
        AscendC::LocalTensor<T> const& src1,
        Inputs const&... rest
    ) const {
        AscendC::Mul(dst, src0, src1, compute_length);
        if constexpr (sizeof...(rest) > 0) {
            AscendC::PipeBarrier<PIPE_V>();
            operator()(dst, compute_length, dst, rest...);
        }
    }
};

template <typename T>
struct Add {
    template <typename... Inputs>
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<T>& dst,
        uint32_t compute_length,
        AscendC::LocalTensor<T> const& src0,
        AscendC::LocalTensor<T> const& src1,
        Inputs const&... rest
    ) const {
        AscendC::Add(dst, src0, src1, compute_length);
        if constexpr (sizeof...(rest) > 0) {
            AscendC::PipeBarrier<PIPE_V>();
            operator()(dst, compute_length, dst, rest...);
        }
    }
};

template <typename T>
struct Sub {
    template <typename... Inputs>
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<T>& dst,
        uint32_t compute_length,
        AscendC::LocalTensor<T> const& src0,
        AscendC::LocalTensor<T> const& src1,
        Inputs const&... rest
    ) const {
        AscendC::Sub(dst, src0, src1, compute_length);
        if constexpr (sizeof...(rest) > 0) {
            AscendC::PipeBarrier<PIPE_V>();
            operator()(dst, compute_length, dst, rest...);
        }
    }
};

template <typename T>
struct Div {
    template <typename... Inputs>
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<T>& dst,
        uint32_t compute_length,
        AscendC::LocalTensor<T> const& src0,
        AscendC::LocalTensor<T> const& src1,
        Inputs const&... rest
    ) const {
        AscendC::Div(dst, src0, src1, compute_length);
        if constexpr (sizeof...(rest) > 0) {
            AscendC::PipeBarrier<PIPE_V>();
            operator()(dst, compute_length, dst, rest...);
        }
    }
};

template <typename T>
struct Max {
    template <typename... Inputs>
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<T>& dst,
        uint32_t compute_length,
        AscendC::LocalTensor<T> const& src0,
        AscendC::LocalTensor<T> const& src1,
        Inputs const&... rest
    ) const {
        AscendC::Max(dst, src0, src1, compute_length);
        if constexpr (sizeof...(rest) > 0) {
            AscendC::PipeBarrier<PIPE_V>();
            operator()(dst, compute_length, dst, rest...);
        }
    }   
};

template <typename T>
struct Min {
    template <typename... Inputs>
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<T>& dst,
        uint32_t compute_length,
        AscendC::LocalTensor<T> const& src0,
        AscendC::LocalTensor<T> const& src1,
        Inputs const&... rest
    ) const {
        AscendC::Min(dst, src0, src1, compute_length);
        if constexpr (sizeof...(rest) > 0) {
            AscendC::PipeBarrier<PIPE_V>();
            operator()(dst, compute_length, dst, rest...);
        }
    }
};

//其他类op
template <typename T>
struct AddRelu {
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<T>& dst,
        uint32_t compute_length,
        AscendC::LocalTensor<T> const& src0,
        AscendC::LocalTensor<T> const& src1
    ) const {
        AscendC::Add(dst, src0, src1, compute_length);
        AscendC::Relu(dst, dst, compute_length);
    }
};



} // namespace Catlass::Epilogue::Fusion

#endif
