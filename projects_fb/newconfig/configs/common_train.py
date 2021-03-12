# common training configs.
# DefaultTrainer is able to handle them
train = dict(
    output_dir="./output",
    init_checkpoint="detectron2://ImageNetPretrained/MSRA/R-50.pkl",
    max_iter=90000,
    amp=dict(enabled=False),  # options for Automatic Mixed Precision
    ddp=dict(  # options for DistributedDataParallel
        broadcast_buffers=False,
        find_unused_parameters=False,
        check_reduction=False,
        fp16_compression=False,
    ),
    checkpoint_period=5000,
    eval_period=5000,
    log_period=20,
    device="cuda"
    # ...
)
