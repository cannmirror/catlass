from dataclasses import dataclass


@dataclass
class GemmShape:
    m: int
    n: int
    k: int

    def __str__(self):
        return "GemmShape<{m}, {n}, {k}>".format(m=self.m, n=self.n, k=self.k)

@dataclass
class GemmCoord:
    m: int
    n: int
    k: int
