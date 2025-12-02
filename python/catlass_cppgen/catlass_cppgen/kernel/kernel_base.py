from abc import abstractmethod
from enum import Enum
from typing import Any, Dict, Tuple, Optional, Type
from catlass_cppgen.catlass.gemm_coord import GemmCoord, GemmShape

GM_ADDR = int


class KernelBase:
    def __init__(self, *args, **kwargs):
        self.l1_tile_shape, self.l0_tile_shape = self.get_default_tile_shape()

    @abstractmethod
    def get_default_tile_shape(self) -> Tuple[GemmShape, GemmShape]:
        pass

    def get_workspace_size(self) -> int:

        return 0

    def need_workspace(self) -> bool:
        return False

    def tune(
        self,
        l1_tile_shape: Optional[GemmShape] = None,
        l0_tile_shape: Optional[GemmShape] = None,
        dispatch_policy=None,  # reserved
        block_scheduler=None,  # reserved
    ):
        self.l1_tile_shape = l1_tile_shape or self.l1_tile_shape
        self.l0_tile_shape = l0_tile_shape or self.l0_tile_shape

    def codegen(self, render_params: Dict[str, Any]) -> str:
        """使用`render_params`中的参数渲染代码.

        :param render_params: 模板中的替换占位符与对应的值.
        :type render_params: Dict[str, Any]
        :return: 渲染后的代码.
        :rtype: str
        """
        for key, value in render_params.items():
            if hasattr(value, "value"):
                value = value.value
            render_params[key] = str(value)
        return self._TEMPLATE.format(
            **render_params,
        )

    def compile(self) -> None:
        pass

    def run(self, **kwargs):
        pass

    def get_args(self) -> Dict[str, Type]:
        param_line = ""
        for line in self._TEMPLATE.split("\n"):
            if "__global__ __aicore__" in line:
                param_line = line
        
        print(param_line)
        return {
            "problemShape": GemmCoord,
            "ptrA": GM_ADDR,
            "ptrB": GM_ADDR,
            "ptrC": GM_ADDR,
        }
