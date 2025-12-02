from abc import ABC, abstractmethod
from typing import Iterable


class Coord:
    def __init__(self, value: Iterable[int]):
        self.idx = tuple(value)


class Layout(ABC):
    value: str = ""

    def __init__(self, shape: Iterable[int]):
        self.shape = tuple(shape)
        self.stride = tuple(1 for _ in range(len(shape)))

    def get_offset(self, coord: Iterable[int]) -> int:
        pass

    @abstractmethod
    def is_need_padding(self, align: int) -> bool:
        pass


class RowMajor(Layout):
    value = "layout::RowMajor"

    def __init__(self, shape: tuple[int, int]):
        super().__init__(shape)
        self.stride = (shape[1], 1)

    def is_need_padding(self, align: int) -> bool:
        if self.stride[0] < 65536:
            return self.stride[0] % align != 0
        else:
            return True

    def get_padding_layout(self, align: int) -> Layout:
        if self.is_need_padding(align):
            return PaddingRowMajor(
                self.shape[0],
                self.shape[1],
                (self.shape[1] + align - 1) // align * align,
            )
        else:
            return self


class ColumnMajor(Layout):
    value = "layout::ColumnMajor"

    def __init__(self, shape: tuple[int, int]):
        super().__init__(shape)
        self.stride = (1, shape[0])

    def is_need_padding(self, align: int) -> bool:
        if self.stride[0] < 65536:
            return self.stride[0] % align != 0
        else:
            return True

    def get_padding_layout(self, align: int) -> Layout:
        if self.is_need_padding(align):
            return PaddingColumnMajor(
                self.shape[0],
                self.shape[1],
                (self.shape[0] + align - 1) // align * align,
            )
        else:
            return self


class PaddingRowMajor(Layout):
    value = "layout::PaddingRowMajor"


class PaddingColumnMajor(Layout):
    value = "layout::PaddingColumnMajor"


class VectorLayout(Layout):
    value = "layout::VectorLayout"

    def __init__(self, shape: int):
        self.shape = shape
        self.stride = 1


class PrivateLayout(Layout):
    def is_need_padding(self, align: int) -> bool:
        return False


class nZ(PrivateLayout):
    value = "layout::nZ"

    def is_need_padding(self, align: int) -> bool:
        return False


class zN(PrivateLayout):
    value = "layout::zN"

    def is_need_padding(self, align: int) -> bool:
        return False


class zZ(PrivateLayout):
    value = "layout::zZ"

    def is_need_padding(self, align: int) -> bool:
        return False


class nN(PrivateLayout):
    value = "layout::nN"

    def is_need_padding(self, align: int) -> bool:
        return False
