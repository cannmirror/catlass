#ifndef CATLASS_EPILOGUE_FUSION_VISITOR_IMPL_BASE_HPP
#define CATLASS_EPILOGUE_FUSION_VISITOR_IMPL_BASE_HPP

#include "catlass/catlass.hpp"
#include "catlass/status.hpp"
#include "tla/int_tuple.hpp"

namespace Catlass::Epilogue::Fusion {

enum class VisitStage : uint8_t {
    LOAD = 0,      // 执行所有load指令
    COMPUTE = 1,   // 执行所有compute指令
    STORE = 2,     // 执行所有store指令
};

template <class... Ops>
struct VisitorImplBase {
    using Arguments = tla::tuple<typename Ops::Arguments...>;
    using Params = tla::tuple<typename Ops::Params...>;

    template <class ProblemShape>
    CATLASS_HOST_DEVICE static constexpr Params
    to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
        uint8_t* op_workspace = reinterpret_cast<uint8_t*>(workspace);
        return tla::transform_apply(
            tla::tuple<Ops...>{}, args,
            [&](auto&& op_tag, auto const& op_args) {
                using Op = typename tla::remove_cvref_t<decltype(op_tag)>;
                auto ret = Op::to_underlying_arguments(problem_shape, op_args, op_workspace);
                if (op_workspace != nullptr) {
                    size_t sz = Op::get_workspace_size(problem_shape, op_args);
                    op_workspace += (sz + 31) & ~31; // 16 字节对齐
                }
                return ret;
            },
            [](auto&&... op_params) -> tla::tuple<tla::remove_cvref_t<decltype(op_params)>...> { 
                return tla::tuple<tla::remove_cvref_t<decltype(op_params)>...>(op_params...); 
            }
        );
    }

    template <class ProblemShape>
    CATLASS_HOST_DEVICE static bool
    can_implement(ProblemShape const& problem_shape, Arguments const& args) {
        return tla::transform_apply(
            tla::tuple<Ops...>{}, args,
            [&](auto&& op_tag, auto const& op_args) {
                using Op = typename tla::remove_cvref_t<decltype(op_tag)>;
                return Op::can_implement(problem_shape, op_args);
            },
            [](auto&&... ok) { return (true && ... && ok); }
        );
    }

    template <class ProblemShape>
    CATLASS_HOST_DEVICE static size_t
    get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
        return tla::transform_apply(
            tla::tuple<Ops...>{}, args,
            [&](auto&& op_tag, auto const& op_args) {
                using Op = typename tla::remove_cvref_t<decltype(op_tag)>;
                size_t sz = Op::get_workspace_size(problem_shape, op_args);
                return (sz + 31) & ~31; // 16 字节对齐
            },
            [](auto&&... rounded_sizes) { return (size_t{0} + ... + rounded_sizes); }
        );
    }

    // 参考 CUTLASS，为每个节点按树序进行工作区初始化。忽略 stream，仅分配并前进指针。
    template <class ProblemShape>
    CATLASS_HOST_DEVICE static Status
    initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
        Status status = Status::kSuccess;
        uint8_t* op_workspace = reinterpret_cast<uint8_t*>(workspace);

        return tla::transform_apply(
            tla::tuple<Ops...>{}, args,
            [&](auto&& op_tag, auto const& op_args) {
                if (status != Status::kSuccess) {
                    return status;
                }
                using Op = typename tla::remove_cvref_t<decltype(op_tag)>;
                // 若 Op 定义了 initialize_workspace，则调用之
                status = Op::initialize_workspace(problem_shape, op_args, op_workspace);
                // 始终推进工作区指针，保证与 get_workspace_size/to_underlying_arguments 的对齐策略一致
                if (op_workspace != nullptr) {
                    size_t sz = Op::get_workspace_size(problem_shape, op_args);
                    op_workspace += (sz + 31) & ~size_t(31);
                }
                return status;
            },
            // 返回最终状态
            [&](auto const&... /*ignored*/) { return status; }
        );
    }

    CATLASS_HOST_DEVICE
    VisitorImplBase() {}

    CATLASS_HOST_DEVICE
    VisitorImplBase(Params const& params)
        : ops(
            tla::transform_apply(
                tla::tuple<Ops...>{}, params,
                [](auto&& op_tag, auto const& op_params) {
                    using Op = typename tla::remove_cvref_t<decltype(op_tag)>;
                    return Op(op_params);
                },
                [](auto&&... built_ops) -> tla::tuple<tla::remove_cvref_t<decltype(built_ops)>...> { 
                    return tla::tuple<tla::remove_cvref_t<decltype(built_ops)>...>(built_ops...); 
                }
            )
        )
    {}

    // 使用 tla::tuple 存储
    tla::tuple<Ops...> ops;
};

} // namespace Catlass::Epilogue::Fusion

#endif
