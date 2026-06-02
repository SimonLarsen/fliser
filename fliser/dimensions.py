from typing import TypeAlias, Callable
from collections.abc import Sequence
from types import EllipsisType
import operator
import math


Size2: TypeAlias = tuple[int, int]
"""2D size tuple."""


class Tile:
    size: Size2
    offset: Size2

    def __init__(self, size: Size2, offset: Size2):
        self.size = (int(size[0]), int(size[1]))
        self.offset = (int(offset[0]), int(offset[1]))

    def slice(self) -> tuple[EllipsisType, slice, slice]:
        """
        Get tensor slice for tile.

        Assumes sliced tensor has shape [..., H, W].

        Examples
        --------
        ```python
        x = torch.rand(2, 3, 768, 1024)
        tile = Tile((128, 128), (32, 64))
        y = x[tile.slice()]
        print(y.shape)
        # torch.size([2, 3, 128, 128])
        ```
        """
        return (
            ...,
            slice(self.offset[0], self.offset[0] + self.size[0]),
            slice(self.offset[1], self.offset[1] + self.size[1]),
        )

    def _apply_op(self, op: Callable[[int, int], int], o: int) -> "Tile":
        new_size = (op(self.size[0], o), op(self.size[1], o))
        new_offset = (op(self.offset[0], o), op(self.offset[1], o))
        return Tile(new_size, new_offset)

    def __floordiv__(self, o: int) -> "Tile":
        """
        Divide by integer (truncated).

        Example
        -------
        ```python
        tile = Tile((64, 64), (128, 256))
        print(tile // 4)
        # Tile(size=(16, 16), offset=(32, 64))
        ```
        """
        return self._apply_op(operator.floordiv, o)

    def __mul__(self, o: int) -> "Tile":
        """
        Multiply by integer.

        Example
        -------
        ```python
        tile = Tile((16, 16), (32, 64))
        print(tile * 4)
        # Tile(size=(64, 64), offset=(128, 256))
        ```
        """
        return self._apply_op(operator.mul, o)

    def __repr__(self) -> str:
        return f"Tile(size={self.size}, offset={self.offset})"


def compute_tile_count(
    image_size: Size2,
    tile_size: Size2,
    min_overlap: int,
) -> Size2:
    """
    Compute number of tiles for a given tile and overlap.

    Parameters
    ----------
    image_size
        Image size (height, width).
    tile_size
        Tile size (height, width).
    min_overlap
        Minimum tile overlap in pixels.

    Returns
    -------
    tuple[int, int]
        Number of tiles (rows, columns).
    """
    height, width = image_size
    tile_height, tile_width = tile_size
    num_tiles_y = math.ceil(
        (height - min_overlap) / (tile_height - min_overlap)
    )
    num_tiles_x = math.ceil((width - min_overlap) / (tile_width - min_overlap))
    return num_tiles_y, num_tiles_x


def tile_size_from_aspect(
    num_pixels: int,
    aspect_ratio: float,
    divisor: int | None = 8,
) -> Size2:
    """
    Compute tile dimensions given aspect ratio.

    Parameters
    ----------
    num_pixels
        Approximate number of pixels in tile (height * width).
    aspect_ratio
        Target aspect ratio (width / height).
    divisor
        Make dimensions divisible by this number.
    """
    if divisor is None:
        divisor = 1
    tile_width = round(
        math.sqrt(aspect_ratio) * math.sqrt(num_pixels) / divisor
    ) * divisor
    tile_height = round(tile_width / aspect_ratio / divisor) * divisor
    return tile_height, tile_width


def find_optimal_tile_size(
    image_size: Size2,
    tile_sizes: Sequence[Size2],
    min_overlap: int = 32,
) -> Size2:
    """
    Find optimal tile size from a set of candidate sizes.

    Parameters
    ----------
    image_size
        Image size (height, width).
    tile_sizes
        Candidate tile sizes (height, width) to evaluate.
    min_overlap
        Minimum required tile overlap in pixels.
    """

    best_tile_count = None
    best_tile_size = tile_sizes[0]
    best_ratio = best_tile_size[1] / best_tile_size[0]

    for tile_size in tile_sizes:
        tile_size = (
            min(tile_size[0], image_size[0]),
            min(tile_size[1], image_size[1]),
        )
        ratio = tile_size[1] / tile_size[0]

        num_tiles_y, num_tiles_x = compute_tile_count(
            image_size, tile_size, min_overlap
        )
        tile_count = num_tiles_y * num_tiles_x

        if (
            best_tile_count is None or
            tile_count < best_tile_count or
            (
                tile_count == best_tile_count and
                abs(best_ratio - 1.0) < abs(ratio - 1.0)
            )
        ):
            best_tile_count = tile_count
            best_tile_size = tile_size
            best_ratio = tile_size[1] / tile_size[0]

    return best_tile_size


def find_optimal_aspect_ratio(
    image_size: Size2,
    tile_num_pixels: int,
    aspect_ratios: Sequence[float] = (1.0,),
    min_overlap: int = 32,
    divisor: int | None = 8,
) -> Size2:
    """
    Find optimal tile size from a set of candidate aspect ratios.

    Parameters
    ----------
    image_size
        Image size (height, width).
    tile_num_pixels
        Approximate number of pixels in tile (height * width).
    aspect_ratios
        Candidate aspect ratios (width / height) to evaluate.
    min_overlap
        Minimum required tile overlap in pixels.
    divisor
        Make dimensions divisible by this number.
    """

    if divisor is None:
        divisor = 1

    tile_sizes: list[Size2] = []
    for ratio in aspect_ratios:
        tile_size = tile_size_from_aspect(tile_num_pixels, ratio, divisor)
        tile_sizes.append(tile_size)

    return find_optimal_tile_size(
        image_size=image_size,
        tile_sizes=tile_sizes,
        min_overlap=min_overlap,
    )
