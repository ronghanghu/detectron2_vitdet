import logging
import os
import sys
from termcolor import colored


class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name")
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name + ".", "")
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


def setup_logger(save_dir=None, distributed_rank=0, color=True, name="detectron2"):
    """
    Args:
        save_dir (str): a directory to save log. If None, will not save log file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results to files for non-master processes
    # TODO can create separate log files for subprocesses to facilitate debugging
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )
    if color:
        formatter = _ColorfulFormatter(
            colored("[%(asctime)s] %(name)s: ", "green") + "%(message)s",
            datefmt="%m/%d %H:%M:%S",
            root_name=name,
        )
    else:
        formatter = plain_formatter
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, "log.txt"))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)

    return logger
