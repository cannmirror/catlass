#include <acl/acl.h>
#include <opdev/common_types.h>
#include "catlass/layout/layout.hpp"
template <class Layout, cla>
int CreateAclTensorFromDataAndLayout(const uint8_t *dataPtr, Catlass::layout::ColumnMajor layout, aclTensor **tensor){
    *tensor=aclCreateTensor(dataPtr, layout.Capacity(), )
}