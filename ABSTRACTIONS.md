## Abstractions
The main abstractions introduced by `maskrcnn_benchmark` that are useful to
have in mind are the following:

### ImageList
In PyTorch, the first dimension of the input to the network generally represents
the batch dimension, and thus all elements of the same batch have the same
height / width.
In order to support images with different sizes and aspect ratios in the same
batch, we created the `ImageList` class, which holds internally a batch of
images (os possibly different sizes). The images are padded with zeros such that
they have the same final size and batched over the first dimension. The original
sizes of the images before padding are stored in the `image_sizes` attribute,
and the batched tensor in `tensors`.
We provide a convenience function `to_image_list` that accepts a few different
input types, including a list of tensors, and returns an `ImageList` object.

```python
from maskrcnn_benchmark.structures.image_list import to_image_list

images = [torch.rand(3, 100, 200), torch.rand(3, 150, 170)]
batched_images = to_image_list(images)

# it is also possible to make the final batched image be a multiple of a number
batched_images_32 = to_image_list(images, size_divisible=32)
```

