# Using and Writing Models

Models (and their sub-models) in detectron2 are built by
functions such as `build_model`, `build_backbone`, `build_roi_heads`:
```python
from detectron2.modeling import build_model
model = build_model(cfg)  # returns a torch.nn.Module
```

You can replace it completely by your own model,
but this is often not practical due to the complexity of a
whole detection model. Therefore, we also provides a registration mechanism,
so you can overwrite the behavior of a sub-model.

For example, to add a new backbone, import this code:
```python
from detectron2.modeling import BACKBONE_REGISTRY, Backbone
@BACKBONE_REGISTRY.register()
class NewBackBone(Backbone):
  def __init__(self, cfg, input_shape):
    # create your own backbone
```
which will allow you to use `cfg.MODEL.BACKBONE.NAME = 'NewBackBone'` in your config file.

To add new abilities to the ROI heads in a generalized R-CNN,
implement a new
[ROIHeads](../modules/modeling.html#detectron2.modeling.ROIHeads)
in the `ROI_HEADS_REGISTRY`. 
See [densepose in detectron2](https://github.com/facebookresearch/detectron2/tree/master/projects/DensePose)
for an example.

Other registries can be found in [API documentation](../modules/modeling.html).
