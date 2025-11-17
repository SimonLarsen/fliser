from typing import cast, Tuple
from enum import Enum
import torch


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
    tile_size: Tuple[int, int],
    overlap: int,
    **kwargs,
) -> torch.FloatTensor:
    """
    Get blending mask of type `mask_type`.

    Parameters
    ----------
    mask_type
        Blending mask type.
    tile_size
        Tile size (height, width).
    overlap
        Tile overlap in pixels.
    """
    if mask_type == MaskType.LINEAR:
        return linear_mask(tile_size, overlap, **kwargs)
    if mask_type == MaskType.SINE:
        return sine_mask(tile_size, overlap, **kwargs)
    raise ValueError(f"Unknown mask type {mask_type}")


def linear_mask(tile_size: Tuple[int, int], overlap: int) -> torch.FloatTensor:
    """
    Get linear blending mask.

    Parameters
    ----------
    tile_size
        Tile size (height, width).
    overlap
        Tile overlap in pixels.
    """
    tile_height, tile_width = tile_size
    overlap_y = min(overlap, tile_height)
    overlap_x = min(overlap, tile_width)

    grad_y = (torch.arange(1, tile_height + 1) / overlap_y).clamp(0.0, 1.0)
    grad_y = torch.minimum(grad_y, grad_y.flip(0))

    grad_x = (torch.arange(1, tile_width + 1) / overlap_x).clamp(0.0, 1.0)
    grad_x = torch.minimum(grad_x, grad_x.flip(0))

    mask = torch.minimum(grad_y[:, None], grad_x[None, :])
    return cast(torch.FloatTensor, mask)


def sine_mask(tile_size: Tuple[int, int], overlap: int) -> torch.FloatTensor:
    """
    Get sinusoid blending mask.

    Parameters
    ----------
    tile_size
        Tile size (height, width).
    overlap
        Tile overlap in pixels.
    """
    linear = linear_mask(tile_size, overlap)
    mask = -(torch.cos(torch.pi * linear) - 1) / 2
    return cast(torch.FloatTensor, mask)
