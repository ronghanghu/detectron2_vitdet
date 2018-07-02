# Helper functions for model building

This folder contains helper functions that facilitate creating
detection models.
It has a few resnet building utilities in resnet, which also
contains a set of reasonable default values that are used.

In `generalized_rcnn.py`, we have a base implementation
for detection models that can be used as a starting point
for building your own model.
You generally won't need to modify the base class
`GeneralizedRCNN`, but your might want to create new
Heads for your specific use-case.
