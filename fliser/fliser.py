from typing import cast, Tuple, Sequence, Iterator
from types import EllipsisType
from dataclasses import dataclass
import torch
from fliser.dimensions import (
    find_optimal_tile_size,
    compute_tile_count,
)
from fliser.masks import MaskType, get_mask


@dataclass
class Tile:
    size: Tuple[int, int]
    offset: Tuple[int, int]

    def slice(self) -> Tuple[EllipsisType, slice, slice]:
        return (
            ...,
            slice(self.offset[0], self.offset[0] + self.size[0]),
            slice(self.offset[1], self.offset[1] + self.size[1]),
        )


class TileIterator:
    _image_size: Tuple[int, int]
    _tile_size: Tuple[int, int]
    _num_tiles: Tuple[int, int]

    def __init__(
        self,
        image_size: Tuple[int, int],
        tile_size: Tuple[int, int],
        num_tiles: Tuple[int, int],
    ):
        self._image_size = image_size
        self._tile_size = tile_size
        self._num_tiles = num_tiles
        self._iy, self._ix = 0, 0

    def __iter__(self) -> "TileIterator":
        return self

    def __next__(self) -> Tile:
        if self._iy >= self._num_tiles[0]:
            raise StopIteration

        image_height, image_width = self._image_size
        tile_height, tile_width = self._tile_size
        num_tiles_y, num_tiles_x = self._num_tiles

        offset_y = 0
        if num_tiles_y > 1:
            offset_y = round(
                self._iy * (image_height - tile_height) / (num_tiles_y - 1)
            )

        offset_x = 0
        if num_tiles_x > 1:
            offset_x = round(
                self._ix * (image_width - tile_width) / (num_tiles_x - 1)
            )

        self._ix += 1
        if self._ix == self._num_tiles[1]:
            self._ix = 0
            self._iy += 1

        return Tile(self._tile_size, (offset_y, offset_x))


class Fliser:
    """Fliser tiled inference helper."""

    _image_size: Tuple[int, int]
    _tile_size: Tuple[int, int]
    _num_tiles: Tuple[int, int]

    _device: torch.device
    _dtype: torch.dtype

    _value_buffer: torch.FloatTensor
    _weight_buffer: torch.FloatTensor

    def __init__(
        self,
        image_size: Tuple[int, int],
        num_channels: int,
        tile_num_pixels: int,
        aspect_ratios: Sequence[float],
        min_overlap: int = 64,
        divisor: int = 8,
        mask_type: MaskType = MaskType.LINEAR,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        """
        Create Fliser.

        Parameters
        ----------
        image_size
            Full inference image size (height, width).
        num_channels
            Number of channels in output image.
        tile_num_pixels
            Approximate number of pixels per tile (height * width).
        aspect_ratios
            Candidate aspect ratios (width / height).
        min_overlap
            Minimum tile overlap in pixels.
        divisor
            Make tile dimensions divisible by this number.
        mask_type
            Mask blending type.
        device
            Torch device to store buffers on.
        dtype
            Torch dtype to store buffers in.
        """
        self._image_size = (image_size[0], image_size[1])
        self._num_channels = num_channels
        self._min_overlap = min_overlap
        self._mask_type = mask_type
        self._device = device
        self._dtype = dtype

        self._tile_size = find_optimal_tile_size(
            image_size=self._image_size,
            tile_num_pixels=tile_num_pixels,
            aspect_ratios=aspect_ratios,
            min_overlap=min_overlap,
            divisor=divisor,
        )
        self._num_tiles = compute_tile_count(
            self._image_size, self._tile_size, min_overlap
        )

        self._value_buffer = cast(
            torch.FloatTensor,
            torch.empty(
                size=(self._num_channels,) + self._image_size,
                dtype=self._dtype,
                device=self._device,
            )
        )
        self._weight_buffer = cast(
            torch.FloatTensor,
            torch.empty(
                size=(1,) + self._image_size,
                dtype=self._dtype,
                device=self._device,
            )
        )
        self.reset()

    def reset(self) -> None:
        """Reset internal buffers."""
        self._value_buffer.zero_()
        self._weight_buffer.zero_()

    def update(self, tile: Tile, values: torch.FloatTensor) -> None:
        """
        Update state with tile predictions.

        Parameters
        ----------
        tile
            Tile for predicted area.
        values
            Predicted values.
        """
        values = cast(
            torch.FloatTensor,
            values.to(dtype=self._dtype, device=self._device),
        )

        mask = get_mask(self._mask_type, tile.size, self._min_overlap)
        mask = cast(
            torch.FloatTensor,
            mask.to(dtype=self._dtype, device=self._device),
        )

        tile_height, tile_width = tile.size
        offset_y, offset_x = tile.offset
        tile_slice = (
            ...,
            slice(offset_y, offset_y + tile_height),
            slice(offset_x, offset_x + tile_width),
        )
        self._value_buffer[tile_slice] += values * mask
        self._weight_buffer[tile_slice] += mask

    def compute(self) -> torch.FloatTensor:
        """Compute final output image."""
        output = self._value_buffer / self._weight_buffer
        return cast(torch.FloatTensor, output)

    def tiles(self) -> Iterator[Tile]:
        """Get iterator for image tiles."""
        return TileIterator(
            image_size=self._image_size,
            tile_size=self._tile_size,
            num_tiles=self._num_tiles,
        )
