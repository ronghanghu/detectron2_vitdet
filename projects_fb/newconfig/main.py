import os

from detectron2.engine import default_argument_parser, launch
from detectron2.utils.comm import get_rank
from detectron2.utils.logger import setup_logger

from defaults import DefaultTrainer
from newconfig import ConfigFile, apply_overrides


def main(args):
    cfg = ConfigFile.load(args.config_file)

    # support override
    cfg = apply_overrides(cfg, args.overrides)

    os.makedirs(cfg.train.output_dir, exist_ok=True)  # TODO: call default_setup()
    setup_logger(cfg.train.output_dir, distributed_rank=get_rank(), name="fvcore")
    setup_logger(cfg.train.output_dir, distributed_rank=get_rank())

    if get_rank() == 0:
        # support serialization
        dump_fname = os.path.join(cfg.train.output_dir, "serialized_config.yaml")
        ConfigFile.save(cfg, dump_fname)
        # cfg = ConfigFile.load(dump_fname)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(args.resume)

    if args.eval_only:
        print(trainer.test(cfg, trainer.model))
    else:
        trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--overrides", nargs="+")
    args = parser.parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
