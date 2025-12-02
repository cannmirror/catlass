from enum import Enum
from enum import auto


class ArchTag(Enum):
    A2 = auto()
    A3 = auto()
    A5 = auto()

    def __str__(self):
        return {
            ArchTag.A2: "Arch::AtlasA2",
            ArchTag.A3: "Arch::AtlasA3",
            ArchTag.A5: "Arch::AtlasA5",
        }[self]
