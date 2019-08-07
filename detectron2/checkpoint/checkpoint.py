import logging
import numpy as np
import os
import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint.model_zoo import cache_file
from detectron2.utils.file_io import PathManager


class Checkpointer(object):
    def __init__(
        self,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        cache_on_load=False,
    ):
        """
        Args:
            model (nn.Module):
            optimizer:
            scheduler:
            save_dir (str): a directory to load and save checkpoint.
            save_to_disk (bool): whether to do saving or not. By default, all
                processes will do loading, but only the master process will do
                saving.
        """
        if isinstance(model, (DistributedDataParallel, DataParallel)):
            model = model.module
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        if save_to_disk is None:
            save_to_disk = comm.is_main_process()
        self.save_to_disk = save_to_disk
        self.logger = logging.getLogger(__name__)
        self.cache_on_load = cache_on_load

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

    def load(self, path: str):
        """
        Load from the given checkpoint.

        Args:
            path (str): path or url to the checkpoint. If empty, will not load anything.

        Returns:
            dict: extra data loaded from the checkpoint, other than model, optimizer and scheduler.
        """
        if not path:
            # no checkpoint provided
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info("Loading checkpoint from {}".format(path))
        if not os.path.isfile(path):
            # In case the file does not exist, PathManager is responsible to
            # looking for the file.
            # We let the main process look for the file first, since the file
            # may need to be downloaded/cached.

            # If you run distributed training on non-shared filesystem,
            # this logic may not work for you. But what else can we do?
            # In that case it might be the best to just manually put the file
            # somewhere to use.
            if comm.is_main_process():
                path = PathManager.get_local_path(path)
            comm.synchronize()
            path = PathManager.get_local_path(path)
            assert os.path.isfile(path), "Checkpoint {} not found!".format(path)
        if os.path.isfile(path) and self.cache_on_load:
            cached_f = cache_file(path)
            self.logger.info("File {} cached in {}".format(path, cached_f))
            path = cached_f

        checkpoint = self._load_file(path)
        self._load_model(checkpoint)
        if "optimizer" in checkpoint and self.optimizer:
            self.logger.info("Loading optimizer from {}".format(path))
            self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        if "scheduler" in checkpoint and self.scheduler:
            self.logger.info("Loading scheduler from {}".format(path))
            self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        # return any further checkpoint data
        return checkpoint

    def has_checkpoint(self):
        """
        Returns:
            bool: whether a checkpoint exists in the target directory
        """
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        """
        Returns:
            str: The latest checkpoint file in target directory.
        """
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read().strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            return ""
        return os.path.join(self.save_dir, last_saved)

    def get_all_checkpoint_files(self):
        """
        Returns:
            List: All available checkpoint files (*.pth) in target directory.
        """
        all_model_checkpoints = [
            os.path.join(self.save_dir, file)
            for file in os.listdir(self.save_dir)
            if os.path.isfile(os.path.join(self.save_dir, file)) and file.endswith(".pth")
        ]
        return all_model_checkpoints

    def resume_or_load(self, path: str, *, resume: bool = True):
        """
        If `resume` is True, this method attempts to
        resume from the last checkpoint, if exists.
        Otherwise, load checkpoint from the given path.

        This is useful when restarting an interrupted training job.
        """
        if resume and self.has_checkpoint():
            path = self.get_checkpoint_file()
        return self.load(path)

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

    def _load_model(self, checkpoint):
        checkpoint_state_dict = checkpoint.pop("model")
        self._convert_ndarray_to_tensor(checkpoint_state_dict)

        # if the state_dict comes from a model that was wrapped in a
        # DataParallel or DistributedDataParallel during serialization,
        # remove the "module" prefix before performing the matching
        _strip_prefix_if_present(checkpoint_state_dict, "module.")
        incompatible = self.model.load_state_dict(checkpoint_state_dict, strict=False)
        if incompatible.missing_keys:
            self.logger.warning(
                "Keys in the model but not found in the checkpoint: "
                + ", ".join(incompatible.missing_keys)
            )
        if incompatible.unexpected_keys:
            self.logger.warning(
                "Keys in the checkpoint but not found in the model: "
                + ", ".join(incompatible.unexpected_keys)
            )

    def _convert_ndarray_to_tensor(self, state_dict):
        """
        In-place convert all numpy arrays in the state_dict to torch tensor.

        Args:
            state_dict: a state-dict to be loaded to the model
        """
        # model could be an OrderedDict with _metadata attribute
        # (as returned by Pytorch's state_dict()). We should preserve these properties.
        for k in list(state_dict.keys()):
            v = state_dict[k]
            if not isinstance(v, np.ndarray) and not isinstance(v, torch.Tensor):
                raise ValueError("Unsupported type found in checkpoint! {}: {}".format(k, type(v)))
            if not isinstance(v, torch.Tensor):
                state_dict[k] = torch.from_numpy(v)


class PeriodicCheckpointer(object):
    """
    Save checkpoints periodically.

    When `.step(iteration)` is called, it will execute `checkpointer.save` on
    the given checkpointer, if iteration is a multiple of period or if `max_iter` is reached.
    """

    def __init__(self, checkpointer, period, max_iter=None):
        """
        Args:
            checkpointer (Checkpointer): the checkpointer object used to save checkpoints
            period (int): the period to save checkpoint.
            max_iter (int): maximum number of iterations.
                When it is reached, a checkpoint named "model_final" will be saved.
                Setting it to None will disable this feature.
        """
        self.checkpointer = checkpointer
        self.period = int(period)
        self.max_iter = max_iter

    def step(self, iteration, **kwargs):
        """
        Perform the appropriate action at the given iteration.

        Args:
            iteration (int): the current iteration, ranged in [0, max_iter-1]
            kwargs: extra data to save, same as in :meth:`Checkpointer.save`
        """
        iteration = int(iteration)
        additional_state = {"iteration": iteration}
        additional_state.update(kwargs)
        if (iteration + 1) % self.period == 0:
            self.checkpointer.save("model_{:07d}".format(iteration), **additional_state)
        if iteration >= self.max_iter - 1:
            self.checkpointer.save("model_final", **additional_state)

    def save(self, name, **kwargs):
        """
        Same argument as :meth:`Checkpointer.save`.
        Use this method to manually save checkpoints outside the schedule.
        """
        return self.checkpointer.save(name, **kwargs)


def _strip_prefix_if_present(state_dict, prefix):
    def strip_ordered_dict(dic):
        keys = sorted(dic.keys())
        if not all(key.startswith(prefix) for key in keys):
            return

        for key in list(dic.keys()):
            value = dic.pop(key)
            newkey = key[len(prefix) :]
            dic[newkey] = value

    strip_ordered_dict(state_dict)

    # also strip the prefix in metadata, if any
    try:
        metadata = state_dict._metadata
    except AttributeError:
        pass
    else:
        strip_ordered_dict(metadata)
