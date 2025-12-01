fliser
======

Helper library for doing tiled inference with PyTorch.

## Installation

```
pip install fliser
```

## Examples

Use `Fliser.from_sizes` to choose the best tile size from a set of candidates.
The tile size that requires the least amount of tiles to cover the full image will be chosen.

```python
from fliser import Fliser

device = torch.device("cuda:0")

# Create model
model = MyModel().to(device)

# Read image
image = load_image(...)
image = image.unsqueeze(0).to(device)
_, c, h, w = x.shape

# Create Fliser helper class
fliser = Fliser.from_sizes(
    image_size=(h, w),
    num_channels=c,
    tile_sizes=[(416, 632), (512, 512), (632, 416)]
    min_overlap=32,
)

for tile in fliser.tiles():
    # Do inference for single tile
    with torch.inference_mode():
        pred = model(image[tile.slice()])
    
    # Update Fliser state
    fliser.update(tile, pred)

# Compute combined output
output = fliser.compute()
```

Use `Fliser.from_aspect_ratios` to instead choose tile size from a set of aspect ratios.

```python
fliser = Fliser.from_aspect_ratios(
    image_size=(h, w),
    num_channels=c,
    tile_num_pixels=512 ** 2,
    aspect_ratios=(0.5, 1.0, 2.0),
    min_overlap=32,
)
```
