# Detectron2

There is no doc right now!
See [Detectron2 Wiki/FAQ Quip](https://fb.quip.com/6LBMAtG25YHf) for installation and usage.
Join our [user group](https://fb.workplace.com/groups/277527419809135/) for discussions.

Below are notes that temporarily live here but may be moved to other places later.

## Simple Demo

We provide a simple demo that illustrates how you can use `detectron2` for inference:
```bash
cd demo
# use input files:
python demo.py --config-file ../configs/quick_schedules/e2e_mask_rcnn_R_50_FPN_inference_acc_test.yaml \
  --input /path/to/images*.jpg --output /path/to/output/directory
# use webcam:
python demo.py --config-file ../configs/quick_schedules/e2e_mask_rcnn_R_50_FPN_inference_acc_test.yaml --webcam
```

### Single GPU training

```bash
python /path_to_detectron2/tools/train_net.py --config-file "/path/to/config/file.yaml"
```

## Environment variables

- `DETECTRON2_ENV_MODULE`: Name of a module defining a function called `setup_environment` to call before running any detectron2 code. This function can be used to perform set up steps that are specific to different computing or cluster environments.

## Troubleshooting
If you have issues running or compiling this code, we have compiled a list of common issues in
[TROUBLESHOOTING.md](TROUBLESHOOTING.md). If your issue is not present there, please feel
free to open a new issue.

## Major Compatibility Differences Compared to Detectron (v1)

- The height and width of a box with corners (x1, y1) and (x2, y2) is computed as width = x2 - x1 and height = y2 - y1; in Detectron v1 a "+ 1" was added both height and width. This makes detectron2 incompatible with models trained using Detectron v1.
- For a dataset with K object categories, these categories are assigned labels [0, K - 1]. If a background category is used (e.g., with a classifier that uses softmax), the background category is assigned label K. In Detectron v1 background was assigned label 0 and the object classes labels [1, K].
- RPN anchors are now center-aligned to feature grid points and they are not quantized. If one loads a model trained using the original RPN anchor definition, then `MODEL.RPN.ANCHOR_GENERATOR_NAME` must be set to `OriginalRPNAnchorGenerator`.
