# common training configs.
# DefaultTrainer is able to handle them
train = dict(
    output_dir="./output",
    init_checkpoint="detectron2://ImageNetPretrained/MSRA/R-50.pkl",
    max_iter=90000,
    amp=dict(enabled=False),
    checkpoint_period=5000,
    eval_period=100,
    log_period=20,
    device="cuda"
    # ...
)
