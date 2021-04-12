from detectron2.config import LazyConfig
from detectron2.engine import default_argument_parser, default_setup, launch

from defaults import DefaultTrainer


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)

    default_setup(cfg, args)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(args.resume)

    if args.eval_only:
        print(trainer.test(cfg, trainer.model))
    else:
        trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    args = parser.parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
