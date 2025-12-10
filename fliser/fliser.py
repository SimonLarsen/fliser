from typing import cast, Tuple, Sequence, Iterator, Callable
from types import EllipsisType
import operator
import torch
from fliser.dimensions import (
    Size2,
    compute_tile_count,
    find_optimal_tile_size,
    find_optimal_aspect_ratio,
)
from fliser.masks import MaskType, get_mask


class Tile:
    size: Size2
    offset: Size2

    def __init__(self, size: Size2, offset: Size2):
        self.size = (int(size[0]), int(size[1]))
        self.offset = (int(offset[0]), int(offset[1]))

    def slice(self) -> Tuple[EllipsisType, slice, slice]:
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

    def _apply_op(self, op: Callable, o: int) -> "Tile":
        new_size = tuple(op(e, o) for e in self.size)
        new_offset = tuple(op(e, o) for e in self.offset)
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
        self._iy, self._ix = 0, 0

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
            offset_x = round(
                self._ix * (image_width - tile_width) / (num_tiles_x - 1)
            )

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

    _value_buffer: torch.FloatTensor
    _weight_buffer: torch.FloatTensor

    def __init__(
        self,
        image_size: Size2,
        num_channels: int,
        tile_size: Size2,
        min_overlap: int = 64,
        mask_type: MaskType = MaskType.LINEAR,
        device: torch.device = torch.device("cpu"),
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
        self._mask_type = mask_type
        self._device = device
        self._dtype = dtype

        self._num_tiles = compute_tile_count(
            self._image_size, self._tile_size, self._min_overlap
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

    @classmethod
    def from_aspect_ratios(
        cls,
        image_size: Size2,
        num_channels: int,
        tile_num_pixels: int,
        aspect_ratios: Sequence[float],
        min_overlap: int = 64,
        divisor: int = 8,
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
            mask_type=mask_type,
            device=device,
            dtype=dtype,
        )

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
        if not len(values.shape) in (3, 4):
            raise ValueError("values tensor should have shape ([B,] C, H, W).")

        if len(values.shape) == 4:
            if values.size(0) != 1:
                raise ValueError("update() only supports batch size 1.")

            values = cast(torch.FloatTensor, values.squeeze(0))

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
