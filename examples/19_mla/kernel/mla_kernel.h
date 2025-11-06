#include <acl/acl.h>

void MLADevice(
    uint8_t blockNum,
    aclrtStream stream, 
    aclDataType dataType,
    bool isTp1Spec,
    uint64_t fftsAddr,
    uint8_t *q,
    uint8_t *qRope,
    uint8_t *k,
    uint8_t *kRope,
    uint8_t *blockTables,
    uint8_t *o,
    uint8_t *s,
    uint8_t *p,
    uint8_t *oTmp,
    uint8_t *oUpdate,
    uint8_t *oCoreTmp,
    uint8_t *l,
    uint8_t *tiling
);