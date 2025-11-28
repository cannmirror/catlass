from catlass_cppgen.common.data_type import DataType

def get_default_accumulator(data_type_A: DataType, data_type_B: DataType)->DataType:
    assert data_type_A == data_type_B
    if data_type_A == DataType.FLOAT16 and data_type_B == DataType.FLOAT16:
        return DataType.FLOAT
    if data_type_A == DataType.FLOAT and data_type_B == DataType.FLOAT16:
        return DataType.FLOAT
    if data_type_A == DataType.BF16 and data_type_B == DataType.BF16:
        return DataType.FLOAT
    if data_type_A == DataType.INT8 and data_type_B == DataType.INT8:
        return DataType.INT32