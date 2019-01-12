import sys
import pprint
import logging


def verify_results(cfg, results):
    """
    Args:
        results(dict of dict)

    Returns:
        bool: whether the verification succeeds or not
    """
    ok = True
    expected_results = cfg.TEST.EXPECTED_RESULTS
    for task, metric, expected, tolerance in expected_results:
        actual = results[task][metric]
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
