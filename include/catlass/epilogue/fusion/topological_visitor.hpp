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

        // 访问节点 i：先访问其子节点，按 EdgeTuple 指定顺序收集输出，再调用第 i 个回调
        template <int I, class Seq, class... Args>
        CATLASS_DEVICE auto visit_node(
            MatrixCoord const& tileOffset,
            MatrixCoord const& localTileOffset,
            MatrixCoord const& tileShape,
            uint32_t calCount,
            VisitStage stage,
            Seq /*edges_seq*/,  // 形如 tla::seq<child...>
            Args const&... args
        ) {
            // 收集子节点输出为一个 tla::tuple<ChildOutputs...>
            auto child_outputs = collect_children<I>(
                tileOffset, localTileOffset, tileShape, calCount, stage, Seq{}, args...
            );

            // 将子节点输出按照 Seq 指定顺序展开，调用第 I 个回调
            return call_current<I>(tileOffset, localTileOffset, tileShape, calCount, stage, child_outputs);
        }

        // 收集子节点输出的实现：针对 tla::seq 索引序列展开
        template <int I, int... ChildIs, class... Args>
        CATLASS_DEVICE auto collect_children(
            MatrixCoord const& tileOffset,
            MatrixCoord const& localTileOffset,
            MatrixCoord const& tileShape,
            uint32_t calCount,
            VisitStage stage,
            tla::seq<ChildIs...> /*seq*/,
            Args const&... args
        ) {
            return tla::tuple<decltype(
                this->template visit_node<ChildIs>(
                    tileOffset, localTileOffset, tileShape, calCount, stage,
                    decltype(tla::get<ChildIs>(EdgeTuple{})){}, args...
                )
            )...>(
                this->template visit_node<ChildIs>(
                    tileOffset, localTileOffset, tileShape, calCount, stage,
                    decltype(tla::get<ChildIs>(EdgeTuple{})){}, args...
                )...
            );
        }

        // 展开 child_outputs 元组并调用第 I 个算子
        template <int I, class ChildOutputs, int... Js>
        CATLASS_DEVICE auto call_current_expand(
            MatrixCoord const& tileOffset,
            MatrixCoord const& localTileOffset,
            MatrixCoord const& tileShape,
            uint32_t calCount,
            VisitStage stage,
            ChildOutputs const& child_outputs,
            tla::seq<Js...>
        ) {
            return tla::get<I>(this->callbacks_tuple).visit(
                tileOffset, localTileOffset, tileShape, calCount, stage,
                tla::get<Js>(child_outputs)...
            );
        }

        template <int I, class ChildOutputs>
        CATLASS_DEVICE auto call_current(
            MatrixCoord const& tileOffset,
            MatrixCoord const& localTileOffset,
            MatrixCoord const& tileShape,
            uint32_t calCount,
            VisitStage stage,
            ChildOutputs const& child_outputs
        ) {
            constexpr int Num = tla::tuple_size<ChildOutputs>::value;
            return call_current_expand<I>(
                tileOffset, localTileOffset, tileShape, calCount, stage,
                child_outputs, tla::make_seq<Num>{}
            );
        }

        // 通过 collect_children 直接递归 visit_node，无需单独 dispatch_child

        // 统一入口：以根节点 R-1 开始访问
        template <typename... Args>
        CATLASS_DEVICE auto visit(
            MatrixCoord const& tileOffset,
            MatrixCoord const& localTileOffset,
            MatrixCoord const& tileShape,
            uint32_t calCount,
            VisitStage stage,
            Args const&... args
        ) {
            constexpr int R = sizeof...(Ops);
            using RootEdges = decltype(tla::get<R - 1>(EdgeTuple{}));
            return visit_node<R - 1>(tileOffset, localTileOffset, tileShape, calCount, stage, RootEdges{}, args...);
        }
    };

    template <class ArchTag>
    CATLASS_DEVICE auto get_callbacks(
        Arch::Resource<ArchTag>& resource,
        uint32_t& ub_offset,
        uint32_t compute_length,
        GemmCoord const& blockShapeMNK,
        GemmCoord const& blockCoordMNK,
        MatrixCoord const& subblockShape,
        MatrixCoord const& subblockCoord,
        AscendC::GlobalTensor<half> const& gmSubblockC,
        layout::RowMajor const& layoutSubblockC
    ) {
        auto base_callbacks = this->VisitorImpl<Ops...>::get_callbacks(
            resource, ub_offset, compute_length,
            blockShapeMNK, blockCoordMNK, subblockShape, subblockCoord,
            gmSubblockC, layoutSubblockC
        );
        return Callbacks<decltype(base_callbacks)>(static_cast<decltype(base_callbacks)&&>(base_callbacks));
    }
};

} // namespace Catlass::Epilogue::Fusion

#endif


