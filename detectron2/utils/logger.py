import logging
import os
import sys
import tempfile
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
            prefix = colored("INFO", "green")
        return prefix + " " + log


def setup_logger(save_dir=None, distributed_rank=0, color=True, name="detectron2"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results to files for non-master processes
    if distributed_rank > 0:
        # Dependencies that do straight printing (instead of using the logging
        # module) make for messy output (e.g., pycocotools printing the same
        # information in each distributed worker). Instead we save stdout and
        # stderr to log files for non-rank-zero processes.
        path = os.path.join(
            save_dir if save_dir else tempfile.gettempdir(),
            "detectron2_pid{:d}_rank{:d}".format(os.getpid(), distributed_rank),
        )
        sys.stdout = open(path + ".stdout", "w")
        sys.stderr = open(path + ".stderr", "w")
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
