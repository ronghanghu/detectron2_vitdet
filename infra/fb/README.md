
You'll need the pytorch bugfix in D13209800 to run detectron2.

Optional:
Copy the whole directory to somewhere else in fbcode,
grep 'vision/detectron2' and change those paths.

## Local Development:

1. Build:

```
buck build @mode/dev-nosan \
	-c python.native_link_strategy=separate \
	//experimental/deeplearning/vision/detectron2/tools/...
```

2. Run locally:

```
buck run //experimental/deeplearning/vision/detectron2/tools:train_net -- \
  --num-gpus 4 \
	--config-file configs/e2e_mask_rcnn_R_50_FPN_1x.yaml \
	PATHS_CATALOG infra/fb/paths_catalog.py \
	SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.002
```

NOTE: due to a pytorch bug (https://github.com/pytorch/pytorch/issues/14394),
multi-gpu training may result in orphaned processes on your devgpu,
which you'll need to kill manually.

## Run on fblearn:

```
./infra/fb/launch.sh configs/e2e_mask_rcnn_R_50_FPN_1x.yaml LetItRock
```
