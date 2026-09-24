import torch
from fliser import BlendMode, Fliser, MaskType
from pytest import approx


def test_blending_min():
    fliser = Fliser.from_sizes(
        image_size=(6, 6),
        num_channels=1,
        tile_sizes=[(4, 4)],
        min_overlap=2,
        blend_mode=BlendMode.MIN,
    )

    for i, tile in enumerate(fliser.tiles()):
        x = torch.full((1, *tile.size), i, dtype=torch.float32)
        fliser.update(tile, x)

    output = fliser.compute()

    expected = torch.tensor(
        [
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 1],
            [2, 2, 2, 2, 3, 3],
            [2, 2, 2, 2, 3, 3],
        ],
        dtype=torch.float32,
    ).unsqueeze(0)

    assert output - expected == approx(0.0, abs=1e-4)


def test_blending_max():
    fliser = Fliser.from_sizes(
        image_size=(6, 6),
        num_channels=1,
        tile_sizes=[(4, 4)],
        min_overlap=2,
        blend_mode=BlendMode.MAX,
    )

    for i, tile in enumerate(fliser.tiles()):
        x = torch.full((1, *tile.size), i, dtype=torch.float32)
        fliser.update(tile, x)

    output = fliser.compute()

    expected = torch.tensor(
        [
            [0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
            [2, 2, 3, 3, 3, 3],
            [2, 2, 3, 3, 3, 3],
            [2, 2, 3, 3, 3, 3],
            [2, 2, 3, 3, 3, 3],
        ],
        dtype=torch.float32,
    ).unsqueeze(0)

    assert output - expected == approx(0.0, abs=1e-4)


def test_blending_linear():
    fliser = Fliser.from_sizes(
        image_size=(6, 6),
        num_channels=1,
        tile_sizes=[(4, 4)],
        min_overlap=2,
        blend_mode=BlendMode.MASK,
        mask_type=MaskType.LINEAR,
    )

    for i, tile in enumerate(fliser.tiles()):
        x = torch.full((1, *tile.size), i, dtype=torch.float32)
        fliser.update(tile, x)

    output = fliser.compute()

    expected = torch.tensor(
        [
            [0, 0, 1, 2, 3, 3],
            [0, 0, 1, 2, 3, 3],
            [2, 2, 3, 4, 5, 5],
            [4, 4, 5, 6, 7, 7],
            [6, 6, 7, 8, 9, 9],
            [6, 6, 7, 8, 9, 9],
        ],
        dtype=torch.float32,
    ).unsqueeze(0) / 3.0

    assert output - expected == approx(0.0, abs=1e-4)
