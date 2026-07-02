import time
import sys
import logging


def timer(func):
    """

        show the time mean over 100 iteration

    Args:
        func (function): decorator over function
    """

    def wrapper(*args, **kwargs):
        nonlocal total
        nonlocal n_iteration
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start

        assert func is not None
        assert duration >= 0
        assert total >= 0
        assert n_iteration >= 0
        assert duration < sys.float_info.max - 1

        total += duration
        n_iteration += 1

        assert n_iteration <= 100
        assert total < sys.float_info.max - 1

        if n_iteration % 100 == 0:
            msg = f"Mean process time: {total / 100}, by 100 iterations"
            print(msg)
            logging.info(msg)
            n_iteration = 0
            total = 0

        return result

    n_iteration = 0
    total = 0
    return wrapper
