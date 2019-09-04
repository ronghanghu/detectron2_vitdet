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
python demo.py --config-file ../configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml \
  --input /path/to/images*.jpg --output /path/to/output/directory
# use webcam:
python demo.py --config-file ../configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml --webcam
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
