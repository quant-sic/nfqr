import logging


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
