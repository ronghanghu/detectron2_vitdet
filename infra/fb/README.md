
## Local Development:

1. Build:

buck build @mode/dev-nosan \
	-c python.native_link_strategy=separate \
	//experimental/deeplearning/yuxinwu/detectron2/tools/...

2. Run with 1 GPU:

buck run //experimental/deeplearning/yuxinwu/detectron2/tools:train_net -- \
	--config-file configs/e2e_mask_rcnn_R_50_FPN_1x.yaml \
	PATHS_CATALOG infra/fb/paths_catalog.py \
	SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.002
