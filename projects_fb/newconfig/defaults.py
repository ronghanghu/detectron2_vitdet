import logging
from torch.nn.parallel import DistributedDataParallel

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config.instantiate import instantiate
from detectron2.engine import AMPTrainer, SimpleTrainer, TrainerBase, default_writers, hooks
from detectron2.evaluation import inference_on_dataset
from detectron2.utils import comm
from detectron2.utils.env import TORCH_VERSION
from detectron2.utils.logger import setup_logger


# NOTE: can be merged with the existing DefaultTrainer
class DefaultTrainer(TrainerBase):
    """
    Examples:
    ::
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load()  # load last checkpoint or MODEL.WEIGHTS
        trainer.train()
    Attributes:
        checkpointer (DetectionCheckpointer):
        cfg ():
    """

    def __init__(self, cfg):
        """
        Args:
            cfg: an object with the following attributes:
                model: instantiate to a module
                dataloader.{train,test}: instantiate to dataloaders
                dataloader.evaluator: instantiate to evaluator for test set
                optimizer: instantaite to an optimizer
                lr_multiplier: instantiate to a fvcore scheduler
                train: keys defined in `common_train.py`
        """
        super().__init__()
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        # cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            fp16_compression = cfg.train.ddp.pop("fp16_compression", False)
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], **cfg.train.ddp
            )
            if fp16_compression:
                from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks

                model.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
        self._trainer = (AMPTrainer if cfg.train.amp.enabled else SimpleTrainer)(
            model, data_loader, optimizer
        )

        # Assume no other objects need to be checkpointed.
        # We can later make it checkpoint the stateful hooks
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.train.output_dir,
            optimizer=optimizer,
        )
        self.start_iter = 0
        self.max_iter = cfg.train.max_iter
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.train.output_dir` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.train.init_checkpoint`` will not be used.
        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.train.init_checkpoint` (but will not load other states) and start
        from iteration 0.
        Args:
            resume (bool): whether to do resume or not
        """
        checkpoint = self.checkpointer.resume_or_load(self.cfg.train.init_checkpoint, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).
        if isinstance(self.model, DistributedDataParallel):
            # broadcast loaded data/model from the first rank, because other
            # machines may not have access to the checkpoint file
            if TORCH_VERSION >= (1, 7):
                self.model._sync_params_and_buffers()
            self.start_iter = comm.all_gather(self.start_iter)[0]

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.
        Returns:
            list[HookBase]:
        """

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(self.cfg.lr_multiplier)),
            hooks.PeriodicCheckpointer(self.checkpointer, self.cfg.train.checkpoint_period)
            if comm.is_main_process()
            else None,
        ]

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(
            hooks.EvalHook(self.cfg.train.eval_period, lambda: self.test(self.cfg, self.model))
        )

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=self.cfg.train.log_period))
        return ret

    def build_writers(self):
        """
        Build a list of writers to be used. By default it contains
        writers that write metrics to the screen,
        a json file, and a tensorboard event file respectively.
        If you'd like a different list of writers, you can overwrite it in
        your trainer.
        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.
        """
        return default_writers(self.cfg.train.output_dir, self.max_iter)

    def train(self):
        super().train(self.start_iter, self.max_iter)

    def run_step(self):
        self._trainer.iter = self.iter
        self._trainer.run_step()

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:
        Overwrite it if you'd like a different model.
        """
        model = instantiate(cfg.model)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        model.to(cfg.train.device)
        return model

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Returns:
            torch.optim.Optimizer:
        Overwrite it if you'd like a different optimizer.
        """
        cfg.optimizer.params.model = model
        return instantiate(cfg.optimizer)

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable
        Overwrite it if you'd like a different data loader.
        """
        return instantiate(cfg.dataloader.train)

    @classmethod
    def test(cls, cfg, model):
        """
        Args:
            model (nn.Module):
        Returns:
            dict: a dict of result metrics
        """
        return inference_on_dataset(
            model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
        )


# Access basic attributes from the underlying trainer
for _attr in ["model", "data_loader", "optimizer"]:
    setattr(
        DefaultTrainer,
        _attr,
        property(
            # getter
            lambda self, x=_attr: getattr(self._trainer, x),
            # setter
            lambda self, value, x=_attr: setattr(self._trainer, x, value),
        ),
    )
