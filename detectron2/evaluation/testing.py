import logging
import numpy as np
import pprint
import sys
from collections import OrderedDict


def print_csv_format(results):
    """
    Print main metrics in a format similar to Detectron,
    so that they are easy to copypaste into a spreadsheet.

    Args:
        results (OrderedDict[dict]): task_name -> {metric -> score}
    """
    assert isinstance(results, OrderedDict), results  # unordered results cannot be properly printed
    logger = logging.getLogger(__name__)
    for task, res in results.items():
        # Don't print "AP-category" metrics since they are usually not tracked.
        important_res = [(k, v) for k, v in res.items() if "AP-" not in k]
        logger.info("copypaste: Task: {}".format(task))
        logger.info("copypaste: " + ",".join([k[0] for k in important_res]))
        logger.info("copypaste: " + ",".join(["{0:.4f}".format(k[1]) for k in important_res]))


def verify_results(cfg, results):
    """
    Args:
        results (OrderedDict[dict]): task_name -> {metric -> score}

    Returns:
        bool: whether the verification succeeds or not
    """
    ok = True
    expected_results = cfg.TEST.EXPECTED_RESULTS
    for task, metric, expected, tolerance in expected_results:
        actual = results[task][metric]
        if not np.isfinite(actual):
            ok = False
        diff = abs(actual - expected)
        if diff > tolerance:
            ok = False

    logger = logging.getLogger(__name__)
    if not ok:
        logger.error("Result verification failed!")
        logger.error("Expected Results: " + str(expected_results))
        logger.error("Actual Results: " + pprint.pformat(results))

        # TODO email
        sys.exit(1)
    else:
        logger.info("Results verification passed.")
    return ok
