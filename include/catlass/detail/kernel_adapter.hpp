/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#ifndef CATLASS_DETAIL_KERNEL_ADAPTER_HPP
#define CATLASS_DETAIL_KERNEL_ADAPTER_HPP

#include "catlass/catlass.hpp"

#ifndef KERNEL_TASK_TYPE
#define KERNEL_TASK_TYPE KERNEL_TYPE_MIX_AIC_1_2
#endif

namespace Catlass {
/// Generic Catlass kernel template
template <class Operator>
CATLASS_GLOBAL void KernelAdapter(typename Operator::Params params)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TASK_TYPE);
    Operator op;
    op(params);
}

} // namespace Catlass
#endif
