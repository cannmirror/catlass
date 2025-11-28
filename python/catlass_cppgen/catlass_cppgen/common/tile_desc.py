from dataclasses import dataclass

from catlass_cppgen.catlass.gemm_coord import GemmShape


@dataclass
class TileDescription:
    l1_tile_shape: GemmShape
    l0_tile_shape: GemmShape

    def __init__(self, l1_tile_shape: GemmShape, l0_tile_shape: GemmShape):
        self.l1_tile_shape = l1_tile_shape
        self.l0_tile_shape = l0_tile_shape
