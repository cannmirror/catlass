#include <acl/acl.h>

void FAInferDevice(
    uint8_t blockNum,
    aclrtStream stream,
    aclDataType dataType,
    uint64_t fftsAddr,
    uint8_t *q,
    uint8_t *k,
    uint8_t *v,
    uint8_t *mask,
    uint8_t *blockTables,
    uint8_t *o,
    uint8_t *actualQseqlen,
    uint8_t *actualKvseqlen,
    uint8_t *s,
    uint8_t *p,
    uint8_t *oTemp,
    uint8_t *oUpdate,
    uint8_t *tiling
)