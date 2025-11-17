fliser
======

Helper library for doing tiled inference with PyTorch.

## Installation

```
pip install git+https://github.com/SimonLarsen/fliser.git
```

## Example

```python
from fliser import Fliser

device = torch.device("cuda:0")

# Create model
model = MyModel().to(device)

# Read image
image = load_image(...)
x = image.unsqueeze(0).to(device)
c, h, w = x.shape

# Create Fliser helper class
fliser = Fliser(
    image_size=(h, w),
    num_channels=c,
    tile_num_pixels=512 ** 2,
    aspect_ratios=(1/2, 1.0, 2/1),
    min_overlap=64,
)

# Do inference once per tile
for tile in fliser.tiles():
    with torch.inference_mode():
        x_tile = x[tile.slice()].unsqueeze(0)
        pred = model(x_tile)[0]
    
    fliser.update(tile, pred)

# Compute combined output
output = fliser.compute()
```
