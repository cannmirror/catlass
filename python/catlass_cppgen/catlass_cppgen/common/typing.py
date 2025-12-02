from typing import Union
import torch
import numpy as np

SupportedTensor = Union[torch.Tensor, np.ndarray]
SupportedDataType = Union[torch.dtype, np.dtype]
