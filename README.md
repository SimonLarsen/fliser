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
image = image.unsqueeze(0).to(device)
_, c, h, w = x.shape

# Create Fliser helper class
fliser = Fliser(
    image_size=(h, w),
    num_channels=c,
    tile_num_pixels=512 ** 2,
    aspect_ratios=(1/2, 1.0, 2/1),
    min_overlap=64,
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
