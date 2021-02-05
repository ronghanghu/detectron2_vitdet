import os
from torch.nn.parallel import DistributedDataParallel

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import SimpleTrainer, default_argument_parser, default_writers, hooks, launch
from detectron2.evaluation import inference_on_dataset
from detectron2.solver import LRMultiplier
from detectron2.utils import comm
from detectron2.utils.logger import setup_logger

from newconfig import ConfigFile, apply_overrides, instantiate


def main(args):
    setup_logger()
    cfg = ConfigFile.load(args.config_file)

    # support override
    cfg = apply_overrides(cfg, args.overrides)

    # support serialization
    dump_fname = "serialized.yaml"
    ConfigFile.save(cfg, dump_fname)
    cfg = ConfigFile.load(dump_fname)

    model = instantiate(cfg.model)
    model.cuda()
    print(model)
    if comm.get_world_size() > 1:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )
    cfg.optimizer.params = model.parameters()
    optimizer = instantiate(cfg.optimizer)

    scheduler = LRMultiplier(optimizer, instantiate(cfg.lr_multiplier), args.max_iter)

    dataloader = cfg.dataloader
    trainer = SimpleTrainer(model, instantiate(dataloader.train), optimizer)
    checkpointer = DetectionCheckpointer(
        model,
        args.output_dir,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=scheduler),
            hooks.PeriodicCheckpointer(checkpointer, 5000) if comm.is_main_process() else None,
            hooks.EvalHook(
                100,
                lambda: inference_on_dataset(
                    model, instantiate(dataloader.test), instantiate(dataloader.evaluator)
                ),
            ),
            hooks.PeriodicWriter(
                default_writers(args.output_dir, args.max_iter),
                period=20,
            )
            if comm.is_main_process()
            else None,
        ]
    )
    trainer.train(0, args.max_iter)


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--max-iter", default=90000, type=int)
    parser.add_argument("--overrides", nargs="+")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
