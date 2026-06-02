from enum import Enum
import torch
from torch import Tensor
from fliser.dimensions import Size2, Tile


class MaskType(str, Enum):
    """
    Blending mask type.

    Attributes
    ----------
    LINEAR
        Linear blending mask.
    SINE
        Sinusoid blending mask.
    """

    LINEAR = "linear"
    SINE = "SINE"


def get_mask(
    mask_type: MaskType,
    tile: Tile,
    image_size: Size2,
    overlap: int,
) -> Tensor:
    """
    Get linear blending mask.

    Parameters
    ----------
    tile
        Tile descriptor.
    image_size
        Full image size (height, width).
    overlap
        Tile overlap in pixels.
    """
    if mask_type == MaskType.LINEAR:
        return _linear_mask(tile, image_size, overlap)
    if mask_type == MaskType.SINE:
        return _sine_mask(tile, image_size, overlap)


def _linear_mask(
    tile: Tile,
    image_size: Size2,
    overlap: int,
) -> Tensor:
    tile_height, tile_width = tile.size
    y, x = tile.offset
    height, width = image_size

    overlap_y = min(overlap, tile_height)
    overlap_x = min(overlap, tile_width)

    grad_y = (torch.arange(1, tile_height + 1) / (overlap_y + 1)).clamp(0.0, 1.0)
    grad_y = grad_y[:, None]

    grad_x = (torch.arange(1, tile_width + 1) / (overlap_x + 1)).clamp(0.0, 1.0)
    grad_x = grad_x[None, :]

    mask = torch.ones((tile_height, tile_width))
    if y > 0:
        mask *= grad_y
    if y + tile_height < height:
        mask *= grad_y.flip(0)
    if x > 0:
        mask *= grad_x
    if x + tile_width < width:
        mask *= grad_x.flip(1)

    return mask


def _sine_mask(
    tile: Tile,
    image_size: Size2,
    overlap: int,
) -> Tensor:
    linear = _linear_mask(tile, image_size, overlap)
    mask = -(torch.cos(torch.pi * linear) - 1) / 2
    return mask
