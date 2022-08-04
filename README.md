# PyTorch Tools

`pytorch_tools` is a collection of functions and classes for training and predicting in PyTorch.

## What is Included?

(Mostly for binary segmentation at this point)

- Losses. Common loss functions used for training models (like Dice).
- Metrics. Functions to compute metrics (like IoU) and a meter class to keep track of them.
- Tiling. Functions por padding and tiling images as well as functions for predicting a complete mask based on a tiled image.
- Training. Functions for training and validation of models.
- Utilities. Additional tools commonly required while training (e.g. RLE, splitting datasets, etc.)

Requirements:
- numpy
- numba
- torch
- tqdm
