from typing import Optional, Sequence, Tuple
import math


def compute_tile_count(
    image_size: Tuple[int, int],
    tile_size: Tuple[int, int],
    min_overlap: int,
) -> Tuple[int, int]:
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
    divisor: Optional[int] = 8,
) -> Tuple[int, int]:
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
    image_size: Tuple[int, int],
    tile_num_pixels: int,
    aspect_ratios: Sequence[float] = (1.0,),
    min_overlap: int = 32,
    divisor: Optional[int] = 8,
) -> Tuple[int, int]:
    """
    Find optimal tile size from a set of candidate aspect ratios.

    image_size
        Image size (height, width).
    tile_num_pixels
        Approximate number of pixels in tile (height * width).
    aspect_ratios
        Candidate aspect ratios.
    min_overlap
        Minimum tile overlap in pixels.
    divisor
        Make dimensions divisible by this number.
    """
    if divisor is None:
        divisor = 1

    best_tile_count = None
    best_ratio = 1.0

    for ratio in aspect_ratios:
        tile_size = tile_size_from_aspect(tile_num_pixels, ratio, divisor)
        tile_size = (
            min(tile_size[0], image_size[0]),
            min(tile_size[1], image_size[1]),
        )

        num_tiles_y, num_tiles_x = compute_tile_count(
            image_size, tile_size, min_overlap
        )
        tile_count = num_tiles_y * num_tiles_x

        if (
            best_tile_count is None or
            tile_count < best_tile_count or
            (
                tile_count == best_tile_count and
                abs(ratio - 1.0) < abs(best_ratio - 1.0)
            )
        ):
            best_tile_count = tile_count
            best_ratio = ratio

    return tile_size_from_aspect(tile_num_pixels, best_ratio, divisor)
