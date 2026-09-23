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
    device: torch.device,
    dtype: torch.dtype,
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
        return _linear_mask(tile, image_size, overlap, device, dtype)
    if mask_type == MaskType.SINE:
        return _sine_mask(tile, image_size, overlap, device, dtype)


def _linear_mask(
    tile: Tile,
    image_size: Size2,
    overlap: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    tile_height, tile_width = tile.size
    y, x = tile.offset
    height, width = image_size

    overlap_y = min(overlap, tile_height)
    overlap_x = min(overlap, tile_width)

    grad_y = (
        torch.linspace(
            1.0 / (overlap_y + 1),
            tile_height / overlap_y,
            tile_height,
            device=device,
            dtype=dtype,
        )
        .clamp_max(1.0)
        .unsqueeze(-1)
    )

    grad_x = (
        torch.linspace(
            1.0 / (overlap_x + 1),
            tile_width / overlap_x,
            tile_width,
            device=device,
            dtype=dtype,
        )
        .clamp_max(1.0)
        .unsqueeze(0)
    )

    mask = torch.ones((1, 1), device=device, dtype=dtype)
    if y > 0:
        mask = mask * grad_y
    if y + tile_height < height:
        mask = mask * grad_y.flip(0)
    if x > 0:
        mask = mask * grad_x
    if x + tile_width < width:
        mask = mask * grad_x.flip(1)

    return mask


def _sine_mask(
    tile: Tile,
    image_size: Size2,
    overlap: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    linear = _linear_mask(tile, image_size, overlap, device, dtype)
    mask = -(torch.cos(torch.pi * linear) - 1) / 2
    return mask
