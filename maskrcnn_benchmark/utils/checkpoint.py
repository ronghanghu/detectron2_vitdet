import logging
import os

import torch

import maskrcnn_benchmark.utils.comm as comm
from maskrcnn_benchmark.utils.model_serialization import load_state_dict
from maskrcnn_benchmark.utils.model_zoo import cache_url


class Checkpointer(object):
    def __init__(self, model, optimizer=None, scheduler=None, save_dir="", save_to_disk=None):
        """
        Args:
            model (nn.Module):
            optimizer:
            scheduler:
            save_dir (str): a directory to load and save checkpoint. TODO: renamed todir
            save_to_disk (bool): whether to do saving or not. By default, all
                processes will do loading, but only the master process will do
                saving.
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        if save_to_disk is None:
            save_to_disk = comm.is_main_process()
        self.save_to_disk = save_to_disk
        self.logger = logging.getLogger(__name__)

    def save(self, name, **kwargs):
        """
        kwargs: extra data to save, in addition to model, optimizer and scheduler
        """
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return

        data = {}
        data["model"] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        basename = "{}.pth".format(name)
        save_file = os.path.join(self.save_dir, basename)
        assert os.path.basename(save_file) == basename, basename
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)
        self.tag_last_checkpoint(basename)

    def load(self, f=None):
        """
        Load from latest checkpoint.
        When a checkpoint does not exist, load from the provided file.

        Returns:
            dict: extra data loaded from the checkpoint, other than model, optimizer and scheduler.
        """
        if self.has_checkpoint():
            # override argument with existing checkpoint
            f = self.get_checkpoint_file()
        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info("Loading checkpoint from {}".format(f))
        if not os.path.isfile(f):
            f = self._download_file(f)
            assert os.path.isfile(f), "Checkpoint {} not found!".format(f)

        checkpoint = self._load_file(f)
        self._load_model(checkpoint)
        if "optimizer" in checkpoint and self.optimizer:
            self.logger.info("Loading optimizer from {}".format(f))
            self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        if "scheduler" in checkpoint and self.scheduler:
            self.logger.info("Loading scheduler from {}".format(f))
            self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        # return any further checkpoint data
        return checkpoint

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read().strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return os.path.join(self.save_dir, last_saved)

    def tag_last_checkpoint(self, last_filename_basename):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename_basename)

    def _load_file(self, f):
        """
        Load a checkpoint file. Can be overwritten by subclasses
        to support different formats.

        Args:
            f (str): a file name
        Returns:
            dict: with keys "model" and optionally others that are saved by the checkpointer
                dict["model"] must be a dict which maps strings to torch.Tensor or numpy arrays.
        """
        return torch.load(f, map_location=torch.device("cpu"))

    def _download_file(self, f):
        """
        Called when the file does not exist.
        Can be overwritten by subclass.

        Args:
            f (str):
        Returns:
            str: a file name
        """
        if os.path.isfile(f):
            return f
        # download url files
        if f.startswith("http"):
            # if the file is a url path, download it and cache it
            cached_f = cache_url(f)
            self.logger.info("url {} cached in {}".format(f, cached_f))
            f = cached_f
        return f

    def _load_model(self, checkpoint):
        model = checkpoint.pop("model")
        model = {
            k: v if isinstance(v, torch.Tensor) else torch.from_numpy(v) for k, v in model.items()
        }
        load_state_dict(self.model, model)
