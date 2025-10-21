#ifndef CATLASS_EPILOGUE_FUSION_TOPOLOGICAL_VISITOR_HPP
#define CATLASS_EPILOGUE_FUSION_TOPOLOGICAL_VISITOR_HPP

#include "catlass/epilogue/fusion/visitor_impl.hpp"

namespace Catlass::Epilogue::Fusion {

// 拓扑顺序访问器：按照 EdgeTuple 指定的依赖顺序访问各节点。
// EdgeTuple 形如 tla::tuple<tla::seq<child_idx...>, ...>，长度等于 Ops 数量，
// 约定最后一个 Op（索引 R-1）为根节点（最终算子）。
template <class EdgeTuple, class... Ops>
struct TopologicalVisitor : VisitorImpl<Ops...> {
    using VisitorImpl<Ops...>::VisitorImpl;

    template<class CallbacksImpl>
    struct Callbacks : CallbacksImpl {
        CATLASS_DEVICE
        Callbacks(CallbacksImpl&& impl)
            : CallbacksImpl(static_cast<CallbacksImpl&&>(impl)) {}

        using CallbacksImpl::callbacks_tuple;

        // 缓存：按节点索引缓存所属阶段的输出，仅在节点 OUTPUT_STAGE 启用
        // 说明：AscendC::LocalTensor 是轻量句柄，这里仅保存引用（在 UB 生命周期内有效）
        tla::tuple<AscendC::LocalTensor<typename Ops::ElementOutput>...> cache_;
        bool visited_[sizeof...(Ops)] = {false};

        // 重置访问标记（在每个 tile 开始时调用）
        template <int... Is>
        CATLASS_DEVICE void reset_flags_impl(tla::seq<Is...>) {
            // 展开赋值
            auto _ = {(visited_[Is] = false, 0)...};
            (void)_; // 避免未使用警告
        }
        CATLASS_DEVICE void reset_flags() {
            reset_flags_impl(tla::make_seq<sizeof...(Ops)>{});
        }

        // 在 LOAD 阶段的第一次访问前清理缓存标记

        // 元函数：获取第 I 个 Op 类型
        template <int I, class T0, class... Ts>
        struct TypeAtHelper { using type = typename TypeAtHelper<I - 1, Ts...>::type; };
        template <class T0, class... Ts>
        struct TypeAtHelper<0, T0, Ts...> { using type = T0; };

        // 访问节点 i：先访问其子节点，按 EdgeTuple 指定顺序收集输出，再调用第 i 个回调
        template <int I, class Seq, class... Args>
        CATLASS_DEVICE auto visit_node(
            MatrixCoord const& tileOffset,
            MatrixCoord const& localTileOffset,
            MatrixCoord const& actualTileShape,
            MatrixCoord const& alignedTileShape,
            uint32_t calCount,
            VisitStage stage,
            Seq /*edges_seq*/,  // 形如 tla::seq<child...>
            Args const&... args
        ) {
            // 若节点达到其输出阶段，尝试命中缓存
            using Op = typename TypeAtHelper<I, Ops...>::type;
            constexpr VisitStage OutStage = Op::OUTPUT_STAGE;

            if constexpr (OutStage != VisitStage::STORE) {
                if (stage == OutStage && visited_[I]) {
                    return tla::get<I>(cache_);
                }
            }

            // 收集子节点输出为一个 tla::tuple<ChildOutputs...>
            auto child_outputs = collect_children<I>(
                tileOffset, localTileOffset, actualTileShape, alignedTileShape, calCount, stage, Seq{}, args...
            );

            // 将子节点输出按照 Seq 指定顺序展开，调用第 I 个回调
            auto ret = call_current<I>(tileOffset, localTileOffset, actualTileShape, alignedTileShape, calCount, stage, child_outputs);

            // 仅在输出阶段缓存（STORE 阶段不缓存以保留副作用）
            if constexpr (OutStage != VisitStage::STORE) {
                if (stage == OutStage) {
                    tla::get<I>(cache_) = ret;
                    visited_[I] = true;
                }
            }

            return ret;
        }

        // 收集子节点输出的实现：针对 tla::seq 索引序列展开
        template <int I, int... ChildIs, class... Args>
        CATLASS_DEVICE auto collect_children(
            MatrixCoord const& tileOffset,
            MatrixCoord const& localTileOffset,
            MatrixCoord const& actualTileShape,
            MatrixCoord const& alignedTileShape,
            uint32_t calCount,
            VisitStage stage,
            tla::seq<ChildIs...> /*seq*/,
            Args const&... args
        ) {
            return tla::tuple<decltype(
                this->template visit_node<ChildIs>(
                    tileOffset, localTileOffset, actualTileShape, alignedTileShape, calCount, stage,
                    decltype(tla::get<ChildIs>(EdgeTuple{})){}, args...
                )
            )...>(
                this->template visit_node<ChildIs>(
                    tileOffset, localTileOffset, actualTileShape, alignedTileShape, calCount, stage,
                    decltype(tla::get<ChildIs>(EdgeTuple{})){}, args...
                )...
            );
        }

        // 展开 child_outputs 元组并调用第 I 个算子
        template <int I, class ChildOutputs, int... Js>
        CATLASS_DEVICE auto call_current_expand(
            MatrixCoord const& tileOffset,
            MatrixCoord const& localTileOffset,
            MatrixCoord const& actualTileShape,
            MatrixCoord const& alignedTileShape,
            uint32_t calCount,
            VisitStage stage,
            ChildOutputs const& child_outputs,
            tla::seq<Js...>
        ) {
            return tla::get<I>(this->callbacks_tuple).visit(
                tileOffset, localTileOffset, actualTileShape, alignedTileShape, calCount, stage,
                tla::get<Js>(child_outputs)...
            );
        }

        template <int I, class ChildOutputs>
        CATLASS_DEVICE auto call_current(
            MatrixCoord const& tileOffset,
            MatrixCoord const& localTileOffset,
            MatrixCoord const& actualTileShape,
            MatrixCoord const& alignedTileShape,
            uint32_t calCount,
            VisitStage stage,
            ChildOutputs const& child_outputs
        ) {
            constexpr int Num = tla::tuple_size<ChildOutputs>::value;
            return call_current_expand<I>(
                tileOffset, localTileOffset, actualTileShape, alignedTileShape, calCount, stage,
                child_outputs, tla::make_seq<Num>{}
            );
        }

        // 通过 collect_children 直接递归 visit_node，无需单独 dispatch_child

        // 统一入口：以根节点 R-1 开始访问
        template <typename... Args>
        CATLASS_DEVICE auto visit(
            MatrixCoord const& tileOffset,
            MatrixCoord const& localTileOffset,
            MatrixCoord const& actualTileShape,
            MatrixCoord const& alignedTileShape,
            uint32_t calCount,
            VisitStage stage,
            Args const&... args
        ) {
            // 当进入一个 tile 的 LOAD 阶段时，重置缓存标记，避免跨 tile 干扰
            if (stage == VisitStage::LOAD) {
                reset_flags();
            }
            constexpr int R = sizeof...(Ops);
            using RootEdges = decltype(tla::get<R - 1>(EdgeTuple{}));
            return visit_node<R - 1>(tileOffset, localTileOffset, actualTileShape, alignedTileShape, calCount, stage, RootEdges{}, args...);
        }
    };

    template <class ArchTag>
    CATLASS_DEVICE auto get_callbacks(
        Arch::Resource<ArchTag>& resource,
        uint32_t& ub_offset,
        uint32_t compute_length,
        AscendC::GlobalTensor<half> const& gmSubblockC,
        layout::RowMajor const& layoutSubblockC
    ) {
        auto base_callbacks = this->VisitorImpl<Ops...>::get_callbacks(
            resource, ub_offset, compute_length,
            gmSubblockC, layoutSubblockC
        );
        return Callbacks<decltype(base_callbacks)>(static_cast<decltype(base_callbacks)&&>(base_callbacks));
    }
};

} // namespace Catlass::Epilogue::Fusion

#endif


