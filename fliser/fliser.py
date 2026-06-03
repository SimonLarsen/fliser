from collections.abc import Sequence, Iterator
from enum import Enum
import torch
from torch import Tensor
from fliser.dimensions import (
    Size2,
    Tile,
    compute_tile_count,
    find_optimal_tile_size,
    find_optimal_aspect_ratio,
)
from fliser.masks import MaskType, get_mask


class BlendMode(str, Enum):
    """
    Tile blending mode type.

    Attributes
    ----------
    MASK
        Blend using mask.
    MAX
        Keep maximum value.
    MIN
        Keep minimum value.
    """

    MASK = "mask"
    MAX = "max"
    MIN = "min"


class TileIterator:
    _image_size: Size2
    _tile_size: Size2
    _num_tiles: Size2

    def __init__(
        self,
        image_size: Size2,
        tile_size: Size2,
        num_tiles: Size2,
    ):
        self._image_size = image_size
        self._tile_size = tile_size
        self._num_tiles = num_tiles
        self._iy: int = 0
        self._ix: int = 0

    def __iter__(self) -> "TileIterator":
        return self

    def __len__(self) -> int:
        return self._num_tiles[0] * self._num_tiles[1]

    def __next__(self) -> Tile:
        image_height, image_width = self._image_size
        tile_height, tile_width = self._tile_size
        num_tiles_y, num_tiles_x = self._num_tiles

        if self._iy == num_tiles_y:
            raise StopIteration

        offset_y = 0
        if num_tiles_y > 1:
            offset_y = round(
                self._iy * (image_height - tile_height) / (num_tiles_y - 1)
            )

        offset_x = 0
        if num_tiles_x > 1:
            offset_x = round(self._ix * (image_width - tile_width) / (num_tiles_x - 1))

        self._ix += 1
        if self._ix == num_tiles_x:
            self._ix = 0
            self._iy += 1

        return Tile(self._tile_size, (offset_y, offset_x))


class Fliser:
    """Fliser tiled inference helper."""

    _image_size: Size2
    _tile_size: Size2
    _num_tiles: Size2

    _device: torch.device
    _dtype: torch.dtype

    _value_buffer: Tensor
    _weight_buffer: Tensor

    def __init__(
        self,
        image_size: Size2,
        num_channels: int,
        tile_size: Size2,
        min_overlap: int = 64,
        blend_mode: BlendMode = BlendMode.MASK,
        mask_type: MaskType = MaskType.LINEAR,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Create Fliser helper.

        Parameters
        ----------
        image_size
            Full inference image size (height, width).
        num_channels
            Number of channels in output image.
        tile_size
            Tile size (height, width).
        min_overlap
            Minimum tile overlap in pixels.
        blend_mode
            Tile blending model.
        mask_type
            Mask blending type.
        device
            Torch device to store buffers on.
        dtype
            Torch dtype to store buffers in.
        """

        self._image_size = (image_size[0], image_size[1])
        self._num_channels = num_channels
        self._tile_size = (tile_size[0], tile_size[1])
        self._min_overlap = min_overlap
        self._blend_mode = blend_mode
        self._mask_type = mask_type
        self._device = torch.device(device)
        self._dtype = dtype

        self._num_tiles = compute_tile_count(
            self._image_size, self._tile_size, self._min_overlap
        )

        self._value_buffer = torch.empty(
            size=(self._num_channels,) + self._image_size,
            dtype=self._dtype,
            device=self._device,
        )
        self._weight_buffer = torch.empty(
            size=(1,) + self._image_size,
            dtype=self._dtype,
            device=self._device,
        )
        self.reset()

    @classmethod
    def from_aspect_ratios(
        cls,
        image_size: Size2,
        num_channels: int,
        tile_num_pixels: int,
        aspect_ratios: Sequence[float],
        min_overlap: int = 64,
        divisor: int = 8,
        blend_mode: BlendMode = BlendMode.MASK,
        mask_type: MaskType = MaskType.LINEAR,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> "Fliser":
        """
        Create Fliser helper from at set of candidate aspect ratios.

        Parameters
        ----------
        image_size
            Full inference image size (height, width).
        num_channels
            Number of channels in output image.
        tile_num_pixels
            Approximate number of pixels per tile (height * width).
        aspect_ratios
            Candidate aspect ratios (width / height) to evaluate.
        min_overlap
            Minimum tile overlap in pixels.
        divisor
            Make tile dimensions divisible by this number.
        blend_mode
            Tile blending model.
        mask_type
            Mask blending type.
        device
            Torch device to store buffers on.
        dtype
            Torch dtype to store buffers in.
        """

        tile_size = find_optimal_aspect_ratio(
            image_size=image_size,
            tile_num_pixels=tile_num_pixels,
            aspect_ratios=aspect_ratios,
            min_overlap=min_overlap,
            divisor=divisor,
        )

        return Fliser(
            image_size=image_size,
            num_channels=num_channels,
            tile_size=tile_size,
            min_overlap=min_overlap,
            blend_mode=blend_mode,
            mask_type=mask_type,
            device=device,
            dtype=dtype,
        )

    @classmethod
    def from_sizes(
        cls,
        image_size: Size2,
        num_channels: int,
        tile_sizes: Sequence[Size2],
        min_overlap: int = 64,
        blend_mode: BlendMode = BlendMode.MASK,
        mask_type: MaskType = MaskType.LINEAR,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> "Fliser":
        """
        Create Fliser helper from at set of candidate aspect ratios.

        Parameters
        ----------
        image_size
            Full inference image size (height, width).
        num_channels
            Number of channels in output image.
        tile_sizes
            Candidate tile sizes (width / height) to evaluate.
        min_overlap
            Minimum tile overlap in pixels.
        blend_mode
            Tile blending model.
        mask_type
            Mask blending type.
        device
            Torch device to store buffers on.
        dtype
            Torch dtype to store buffers in.
        """

        tile_size = find_optimal_tile_size(
            image_size=image_size,
            tile_sizes=tile_sizes,
            min_overlap=min_overlap,
        )

        return Fliser(
            image_size=image_size,
            num_channels=num_channels,
            tile_size=tile_size,
            min_overlap=min_overlap,
            blend_mode=blend_mode,
            mask_type=mask_type,
            device=device,
            dtype=dtype,
        )

    def reset(self) -> None:
        """Reset internal buffers."""
        _ = self._value_buffer.zero_()
        _ = self._weight_buffer.zero_()

    def update(self, tile: Tile, values: Tensor) -> None:
        """
        Update state with tile predictions.

        Parameters
        ----------
        tile
            Tile for predicted area.
        values
            Predicted values.
        """
        if not len(values.shape) in (3, 4):
            raise ValueError("values tensor should have shape ([B,] C, H, W).")

        if len(values.shape) == 4:
            if values.size(0) != 1:
                raise ValueError("update() only supports batch size 1.")

            values = values.squeeze(0)

        values = values.to(dtype=self._dtype, device=self._device)

        tile_height, tile_width = tile.size
        offset_y, offset_x = tile.offset
        tile_slice = (
            ...,
            slice(offset_y, offset_y + tile_height),
            slice(offset_x, offset_x + tile_width),
        )

        if self._blend_mode == BlendMode.MASK:
            mask = get_mask(self._mask_type, tile, self._image_size, self._min_overlap)
            mask = mask.to(dtype=self._dtype, device=self._device)
            self._value_buffer[tile_slice] += values * mask
            self._weight_buffer[tile_slice] += mask

        elif self._blend_mode == BlendMode.MAX:
            use_max = self._weight_buffer[tile_slice] > 0.0
            max_values = torch.maximum(self._value_buffer[tile_slice], values)
            self._value_buffer[tile_slice] = torch.where(use_max, max_values, values)
            self._weight_buffer[tile_slice] = 1.0

        elif self._blend_mode == BlendMode.MIN:
            use_min = self._weight_buffer[tile_slice] > 0.0
            min_values = torch.minimum(self._value_buffer[tile_slice], values)
            self._value_buffer[tile_slice] = torch.where(use_min, min_values, values)
            self._weight_buffer[tile_slice] = 1.0

    def compute(self) -> Tensor:
        """Compute final output image."""
        output = self._value_buffer / self._weight_buffer
        return output

    def tiles(self) -> Iterator[Tile]:
        """
        Get iterator for image tiles.

        Returns
        -------
        Iterator[Tile]
            Iterator over all tiles in the image.
        """
        return TileIterator(
            image_size=self._image_size,
            tile_size=self._tile_size,
            num_tiles=self._num_tiles,
        )
