import json
from collections import defaultdict
import torch


_CURRENT_STORAGE = None


def get_event_storage():
    assert (
        _CURRENT_STORAGE is not None
    ), "get_event_storage() has to be called inside a 'with EventStorage(...)' context!"
    return _CURRENT_STORAGE


class HistoryBuffer:
    """
    Track a series of scalar values and provide access to smoothed values over a
    window or the global average of the series.
    """

    def __init__(self):
        self._data = []
        self._count = 0
        self._global_avg = 0

    def update(self, value, iteration):
        """
        Add a new scalar value produced at certain iteration.

        NOTE: The (value, iteration) pair is appended to a list and stored forever.
        Be careful not to abuse it.

        If this turns out to be a memory/perf issue, we can set a limit.
        """
        self._data.append((value, iteration))

        self._count += 1
        self._global_avg += (value - self._global_avg) / self._count

    def latest(self):
        return self._data[-1][0]

    def median(self, window_size):
        d = torch.tensor([x[0] for x in self._data[-window_size:]])
        return d.median().item()

    def avg(self, window_size):
        d = torch.tensor([x[0] for x in self._data[-window_size:]])
        return d.mean().item()

    def global_avg(self):
        return self._global_avg

    def values(self):
        """
        Returns:
            list[(number, iteration)]: all history
        """
        return self._data


class JSONWriter:
    """
    Write scalars to a json file.

    It saves scalars as one json per line (instead of a big json) for easy parsing.

    Examples parsing such a json file:

        $ cat metrics.json | jq -s '.[0:2]'
        [
          {
            "data_time": 0.008433341979980469,
            "iteration": 20,
            "loss": 1.9228371381759644,
            "loss_box_reg": 0.050025828182697296,
            "loss_classifier": 0.5316952466964722,
            "loss_mask": 0.7236229181289673,
            "loss_rpn_box": 0.0856662318110466,
            "loss_rpn_cls": 0.48198649287223816,
            "lr": 0.007173333333333333,
            "time": 0.25401854515075684
          },
          {
            "data_time": 0.007216215133666992,
            "iteration": 40,
            "loss": 1.282649278640747,
            "loss_box_reg": 0.06222952902317047,
            "loss_classifier": 0.30682939291000366,
            "loss_mask": 0.6970193982124329,
            "loss_rpn_box": 0.038663312792778015,
            "loss_rpn_cls": 0.1471673548221588,
            "lr": 0.007706666666666667,
            "time": 0.2490077018737793
          }
        ]

        $ cat metrics.json | jq '.loss_mask'
        0.7126231789588928
        0.689423680305481
        0.6776131987571716
        ...

    """

    def __init__(self, json_file, window_size=20):
        """
        Args:
            json_file (str): path to the json file. New data will be appended if the file exists.
            window_size (int): the window size of median smoothing for the scalars whose
                `smoothing_hint` are True.
        """
        self._file_handle = open(json_file, "a")
        self._window_size = window_size

    def write(self):
        storage = get_event_storage()
        to_save = {"iteration": storage.iteration}
        to_save.update(storage.latest_with_smoothing_hint(self._window_size))
        self._file_handle.write(json.dumps(to_save, sort_keys=True) + "\n")
        self._file_handle.flush()

    def __del__(self):
        # not guaranteed to be called at exit, but probably fine
        self._file_handle.close()


class TensorboardXWriter:
    def __init__(self, log_dir: str, window_size: int = 20, **kwargs):
        """
        Args:
            log_dir (str): The directory to save the output events
            window_size (int): the scalars will be median-smoothed by this window size
            kwargs: other arguments passed to `tensorboardX.SummaryWriter(...)`
        """
        self._window_size = window_size
        import tensorboardX

        self._writer = tensorboardX.SummaryWriter(log_dir, **kwargs)

    def write(self):
        storage = get_event_storage()
        for k, v in storage.latest_with_smoothing_hint(self._window_size).items():
            self._writer.add_scalar(k, v, storage.iteration)

    def __del__(self):
        self._writer.close()


class EventStorage:
    """
    The user-facing class that provides metric storage functionalities.

    In the future we may add support for storing / logging other types of data if needed.
    """

    def __init__(self, start_iter=0):
        """
        Args:
            start_iter (int): the iteration number to start with
        """
        self._history = defaultdict(HistoryBuffer)
        self._smoothing_hints = {}
        self._latest_scalars = {}
        self._iter = start_iter

    def put_scalar(self, name, value, smoothing_hint=True):
        """
        Add a scalar `value` to the `HistoryBuffer` associated with `name`.

        Args:
            smoothing_hint (bool): a 'hint' on whether this scalar is noisy and should be
                smoothed when logged. The hint will be accessible through
                :meth:`EventStorage.smoothing_hints`.  A writer may ignore the hint
                and apply custom smoothing rule.

                It defaults to True because most scalars we save need to be smoothed to
                provide any useful signal.
        """
        history = self._history[name]
        history.update(value, self._iter)
        self._latest_scalars[name] = value

        existing_hint = self._smoothing_hints.get(name)
        if existing_hint is not None:
            assert existing_hint == smoothing_hint
        else:
            self._smoothing_hints[name] = smoothing_hint

    def put_scalars(self, *, smoothing_hint=True, **kwargs):
        """
        Put multiple scalars from keyword arguments.

        Examples:

            storage.put_scalars(loss=my_loss, accuracy=my_accuracy, smoothing_hint=True)
        """
        for k, v in kwargs.items():
            self.put_scalar(k, v, smoothing_hint=smoothing_hint)

    def history(self, name):
        """
        Returns:
            HistoryBuffer: the scalar history for name
        """
        ret = self._history.get(name, None)
        if ret is None:
            raise KeyError("No history metric available for {}!".format(name))
        return ret

    def histories(self):
        """
        Returns:
            dict[name -> HistoryBuffer]: the HistoryBuffer for all scalars
        """
        return self._history

    def latest(self):
        """
        Returns:
            dict[name -> number]: the scalars that's added in the current iteration.
        """
        return self._latest_scalars

    def latest_with_smoothing_hint(self, window_size=20):
        """
        Similar to :meth:`latest`, but the returned values
        are either the un-smoothed original latest value,
        or a median of the given window_size,
        depend on whether the smoothing_hint is True.

        This provides a default behavior that other monitors can use.
        """
        result = {}
        for k, v in self._latest_scalars.items():
            result[k] = self._history[k].median(window_size) if self._smoothing_hints[k] else v
        return result

    def smoothing_hints(self):
        """
        Returns:
            dict[name -> bool]: the user-provided hint on whether the scalar is noisy and needs smoothing.
        """
        return self._smoothing_hints

    def step(self):
        """
        User should call this function at the beginning of each iteration, to
        notify the storage of the start of a new iteration.
        The storage will then be able to associate the new data with the
        correct iteration number.
        """
        self._iter += 1
        self._latest_scalars = {}

    @property
    def iteration(self):
        return self._iter

    def __enter__(self):
        global _CURRENT_STORAGE
        assert _CURRENT_STORAGE is None, "Cannot nest two EventStorage!"
        _CURRENT_STORAGE = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _CURRENT_STORAGE
        _CURRENT_STORAGE = None
