import json
import logging

import numpy as np

def set_par_list_or_dict(list_or_dict,set_fn):

    if isinstance(list_or_dict, (list, dict)):

        if isinstance(list_or_dict, dict):
            iterator = list_or_dict.copy().items()
        elif isinstance(list_or_dict, list):
            iterator = enumerate(list_or_dict.copy())

        for key, value in iterator:
            if isinstance(value, dict):
                list_or_dict[key] = set_par(value,set_fn)

            elif isinstance(value, list):
                list_or_dict[key] = [
                    set_par(list_item,set_fn) for list_item in value
                ]

            list_or_dict = set_fn(key,list_or_dict)

    return list_or_dict

def create_logger(app_name, level: int = logging.INFO) -> logging.Logger:
    """
    Serves as a unified way to instantiate a new logger. Will create a new logging instance with the name app_name. The logging output is sent to the console via a logging.StreamHandler() instance. The output will be formatted using the logging time, the logger name, the level at which the logger was called and the logging message. As the root logger threshold is set to WARNING, the instantiation via logging.getLogger(__name__) results in a logger instance, which console handel also has the threshold set to WARNING. One needs to additionally set the console handler level to the desired level, which is done by this function.

    NOTE: Function might be adapted for more specialized usage in the future

    Args:
        app_name (str): Name of the logger. Will appear in the console output
        level (int): threshold level for the new logger.

    Returns:
        logger: new logging instance

    Examples::

    >>> import logging
    >>> logger=create_logger(__name__,logging.DEBUG)
    """

    # create new up logger
    logger = logging.getLogger(app_name)
    logger.setLevel(level)

    # create console handler and set level
    ch = logging.StreamHandler()
    ch.setLevel(level)

    # create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # add formatter to the console handler
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""

    def default(self, obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {"real": obj.real, "imag": obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)):
            return None

        return json.JSONEncoder.default(self, obj)
