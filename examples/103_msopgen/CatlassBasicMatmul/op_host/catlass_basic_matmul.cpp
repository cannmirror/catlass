
#include <tiling/platform/platform_ascendc.h>

#include "catlass_basic_matmul_tiling.h"
#include "register/op_def_registry.h"

// Macro function for unwinding ge errors.
#define GE_CHECK(status)                                                                                               \
    do {                                                                                                               \
        ge::graphStatus error = status;                                                                                \
        if (error != ge::GRAPH_SUCCESS) {                                                                              \
            std::cerr << __FILE__ << ":" << __LINE__ << " geError:" << error << std::endl;                             \
            return error;                                                                                              \
        }                                                                                                              \
    } while (0)

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{

    CatlassBasicMatmulTilingData tiling;
    const ge::DataType selfDataType = context->GetInputDesc(0)->GetDataType();
    const ge::DataType mat2DataType = context->GetInputDesc(1)->GetDataType();
    if (selfDataType != mat2DataType) {
        std::cerr << "input datatype is different, cannot be multiplied" << std::endl;
        return ge::GRAPH_PARAM_INVALID;
    }
    const gert::StorageShape *self_shape = context->GetInputShape(0);
    const gert::StorageShape *mat2_shape = context->GetInputShape(1);
    const gert::StorageShape *out_shape = context->GetOutputShape(0);

    // check dim_num
    size_t self_dim_num = self_shape->GetOriginShape().GetDimNum();
    if (self_dim_num != 2) {
        std::cerr << "self dim num is not 2" << std::endl;
        return ge::GRAPH_PARAM_INVALID;
    }
    size_t mat2_dim_num = mat2_shape->GetOriginShape().GetDimNum();
    if (mat2_dim_num != 2) {
        std::cerr << "mat2 dim num is not 2" << std::endl;
        return ge::GRAPH_PARAM_INVALID;
    }
    size_t out_dim_num = out_shape->GetOriginShape().GetDimNum();
    if (out_dim_num != 2) {
        std::cerr << "out dim num is not 2" << std::endl;
        return ge::GRAPH_PARAM_INVALID;
    }
    int64_t self_m = self_shape->GetOriginShape().GetDim(0);
    int64_t self_k = self_shape->GetOriginShape().GetDim(1);
    int64_t mat2_k = mat2_shape->GetOriginShape().GetDim(0);
    int64_t mat2_n = mat2_shape->GetOriginShape().GetDim(1);
    int64_t out_m = out_shape->GetOriginShape().GetDim(0);
    int64_t out_n = out_shape->GetOriginShape().GetDim(1);
    if (self_m != out_m) {
        std::cerr << "m is unequal" << std::endl;
        return ge::GRAPH_PARAM_INVALID;
    }
    if (self_k != mat2_k) {
        std::cerr << "k is unequal" << std::endl;
        return ge::GRAPH_PARAM_INVALID;
    }
    if (mat2_n != out_n) {
        std::cerr << "n is unequal" << std::endl;
        return ge::GRAPH_PARAM_INVALID;
    }
    tiling.set_m(self_m);
    tiling.set_n(mat2_n);
    tiling.set_k(self_k);
    context->SetBlockDim(platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic());
    context->SetTilingKey(static_cast<uint64_t>(selfDataType));
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    // if (context->GetInputNum() != 2) {
    //     std::cerr << "input num is not 2";
    //     return ge::GRAPH_PARAM_INVALID;
    // }
    // if (context->GetOutputNum() != 1) {
    //     std::cerr << "output num is not 1";
    //     return ge::GRAPH_PARAM_INVALID;
    // }

    const gert::Shape *self_shape = context->GetInputShape(0);
    const gert::Shape *mat2_shape = context->GetInputShape(1);
    size_t self_dim_num = self_shape->GetDimNum();
    if (self_dim_num != 2) {
        std::cerr << "self dim num is not 2" << std::endl;
        return ge::GRAPH_PARAM_INVALID;
    }
    size_t mat2_dim_num = mat2_shape->GetDimNum();
    if (mat2_dim_num != 2) {
        std::cerr << "mat2 dim num is not 2" << std::endl;
        return ge::GRAPH_PARAM_INVALID;
    }
    int64_t self_m = self_shape->GetDim(0);
    int64_t self_k = self_shape->GetDim(1);
    int64_t mat2_k = mat2_shape->GetDim(0);
    int64_t mat2_n = mat2_shape->GetDim(1);
    if (self_k != mat2_k) {
        std::cerr << "k is unequal" << std::endl;
        return ge::GRAPH_PARAM_INVALID;
    }
    gert::Shape *out_shape = context->GetOutputShape(0);
    out_shape->SetDim(0, self_m);
    out_shape->SetDim(1, mat2_n);
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    // if (context->GetInputNum() != 2) {
    //     std::cerr << "input num is not 2";
    //     return ge::GRAPH_PARAM_INVALID;
    // }
    const ge::DataType selfDataType = context->GetInputDataType(0);
    const ge::DataType mat2DataType = context->GetInputDataType(1);
    if (selfDataType != mat2DataType) {
        std::cerr << "input datatype is different, cannot be multiplied" << std::endl;
        return ge::GRAPH_PARAM_INVALID;
    }
    context->SetOutputDataType(0, selfDataType);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class CatlassBasicMatmul : public OpDef {
public:
    explicit CatlassBasicMatmul(const char *name)
        : OpDef(name)
    {
        this->Input("mat2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("self")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("cubeMathType").Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(CatlassBasicMatmul);
} // namespace ops
