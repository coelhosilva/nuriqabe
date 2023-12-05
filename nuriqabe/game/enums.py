__all__ = [
    "CellCategory",
]

from enum import Enum


class CellCategory(Enum):
    UNKNOWN = -1
    WHITE = 0
    BLACK = 1
