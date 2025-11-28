from enum import Enum
from typing import Union
from catlass_cppgen.common.typing import SupportedDataType
import torch
import numpy as np


class DataType(Enum):
    UNDEFINED = -1
    FLOAT = 0
    FLOAT16 = 1
    INT8 = 2
    INT32 = 3
    UINT8 = 4
    INT16 = 6
    UINT16 = 7
    UINT32 = 8
    INT64 = 9
    UINT64 = 10
    DOUBLE = 11
    BOOL = 12
    STRING = 13
    COMPLEX64 = 16
    COMPLEX128 = 17
    BF16 = 27
    INT4 = 29
    UINT1 = 30
    COMPLEX32 = 33
    HIFLOAT8 = 34
    FLOAT8_E5M2 = 35
    FLOAT8_E4M3FN = 36
    FLOAT8_E8M0 = 37
    FLOAT6_E3M2 = 38
    FLOAT6_E2M3 = 39
    FLOAT4_E2M1 = 40
    FLOAT4_E1M2 = 41

    @classmethod
    def from_dtype(cls, raw_dtype: SupportedDataType) -> "DataType":
        """Convert torch or numpy dtype to DataType enum."""
        # Build dtype mapper
        dtype_mapper = {
            # PyTorch dtypes
            torch.float32: cls.FLOAT,
            torch.float: cls.FLOAT,
            torch.float16: cls.FLOAT16,
            torch.half: cls.FLOAT16,
            torch.float64: cls.DOUBLE,
            torch.double: cls.DOUBLE,
            torch.int8: cls.INT8,
            torch.int16: cls.INT16,
            torch.short: cls.INT16,
            torch.int32: cls.INT32,
            torch.int: cls.INT32,
            torch.int64: cls.INT64,
            torch.long: cls.INT64,
            torch.uint8: cls.UINT8,
            torch.bool: cls.BOOL,
            torch.bfloat16: cls.BF16,
            torch.complex64: cls.COMPLEX64,
            torch.complex128: cls.COMPLEX128,
            # NumPy dtypes
            np.float32: cls.FLOAT,
            np.float16: cls.FLOAT16,
            np.float64: cls.DOUBLE,
            np.int8: cls.INT8,
            np.int16: cls.INT16,
            np.int32: cls.INT32,
            np.int64: cls.INT64,
            np.uint8: cls.UINT8,
            np.uint16: cls.UINT16,
            np.uint32: cls.UINT32,
            np.uint64: cls.UINT64,
            np.bool_: cls.BOOL,
            np.complex64: cls.COMPLEX64,
            np.complex128: cls.COMPLEX128,
        }

        # Add optional PyTorch float8 types if available
        if hasattr(torch, "float8_e5m2"):
            dtype_mapper[torch.float8_e5m2] = cls.FLOAT8_E5M2
        if hasattr(torch, "float8_e4m3fn"):
            dtype_mapper[torch.float8_e4m3fn] = cls.FLOAT8_E4M3FN

        # Try direct lookup first
        if raw_dtype in dtype_mapper:
            return dtype_mapper[raw_dtype]

        # Handle numpy dtype objects
        if isinstance(raw_dtype, np.dtype):
            # Try by type
            dtype_type = raw_dtype.type
            if dtype_type in dtype_mapper:
                return dtype_mapper[dtype_type]
            # Try by name string
            dtype_name = str(raw_dtype)
            # Map common numpy dtype names
            name_mapper = {
                "float32": cls.FLOAT,
                "float16": cls.FLOAT16,
                "float64": cls.DOUBLE,
                "int8": cls.INT8,
                "int16": cls.INT16,
                "int32": cls.INT32,
                "int64": cls.INT64,
                "uint8": cls.UINT8,
                "uint16": cls.UINT16,
                "uint32": cls.UINT32,
                "uint64": cls.UINT64,
                "bool": cls.BOOL,
                "bool_": cls.BOOL,
                "complex64": cls.COMPLEX64,
                "complex128": cls.COMPLEX128,
            }
            if dtype_name in name_mapper:
                return name_mapper[dtype_name]

        # Handle torch dtype objects
        if isinstance(raw_dtype, torch.dtype):
            # Try by string representation
            dtype_str = str(raw_dtype)
            str_mapper = {
                "torch.float32": cls.FLOAT,
                "torch.float": cls.FLOAT,
                "torch.float16": cls.FLOAT16,
                "torch.half": cls.FLOAT16,
                "torch.float64": cls.DOUBLE,
                "torch.double": cls.DOUBLE,
                "torch.int8": cls.INT8,
                "torch.int16": cls.INT16,
                "torch.short": cls.INT16,
                "torch.int32": cls.INT32,
                "torch.int": cls.INT32,
                "torch.int64": cls.INT64,
                "torch.long": cls.INT64,
                "torch.uint8": cls.UINT8,
                "torch.bool": cls.BOOL,
                "torch.bfloat16": cls.BF16,
                "torch.complex64": cls.COMPLEX64,
                "torch.complex128": cls.COMPLEX128,
            }
            if dtype_str in str_mapper:
                return str_mapper[dtype_str]
            # Try optional float8 types
            if dtype_str == "torch.float8_e5m2":
                return cls.FLOAT8_E5M2
            if dtype_str == "torch.float8_e4m3fn":
                return cls.FLOAT8_E4M3FN

        # If not found, return UNDEFINED
        return cls.UNDEFINED
