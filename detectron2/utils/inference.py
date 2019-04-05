import datetime
import logging
import time
from collections import OrderedDict
from contextlib import contextmanager
import torch


class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, input, output):
        """
        Process an input/output pair.

        input: the input that's used to call the model.
        output: the return value of `model(output)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.
        """
        pass


def inference_on_dataset(model, data_loader, evaluator):
    """
    Run model on the data_loader and evaluate the results.

    Args:
        model: a callable which takes an object from `data_loader` and returns some outputs
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run

    Returns:
        The return vaule of `evaluator.evalute()`
    """
    num_devices = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} images".format(len(data_loader)))
    start_time = time.time()

    total = len(data_loader)  # inference data loader must have a fixed length
    evaluator.reset()
    for idx, inputs in enumerate(data_loader):
        with torch.no_grad():
            outputs = model(inputs)
        evaluator.process(inputs, outputs)

        if (idx + 1) % 50 == 0:
            duration = time.time() - start_time
            logger.info(
                "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                    idx + 1,
                    total,
                    duration / (idx + 1),
                    str(datetime.timedelta(seconds=int(duration / (idx + 1) * total - duration))),
                )
            )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = int(time.time() - start_time)
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / len(data_loader), num_devices
        )
    )

    return evaluator.evaluate()


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)


def print_csv_format(results):
    """
    Print results in a format that's easy to copypaste into a spreadsheet.

    Args:
        results: OrderedDict
    """
    assert isinstance(results, OrderedDict)  # unordered results cannot be properly printed
    logger = logging.getLogger(__name__)
    for task in ["bbox", "segm", "keypoints", "box_proposals", "sem_seg"]:
        if task not in results:
            continue
        res = results[task]
        logger.info("copypaste: Task: {}".format(task))
        logger.info("copypaste: " + ",".join([n for n in res.keys()]))
        logger.info("copypaste: " + ",".join(["{0:.4f}".format(v) for v in res.values()]))
