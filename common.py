import logging
import os
from datetime import datetime


def get_logger(name: str, path: str) -> logging.Logger:
    """
    Returns a logger that logs to the specified path.
    :param name:
    :param path:
    :return logging.Logger:
    """

    logger: logging.Logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter: logging.Formatter = logging.Formatter(
        '%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    )

    log_name: str = f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.log'
    if not os.path.exists(f'log/{path}'):
        os.makedirs(f'log/{path}')
    if not os.path.exists(f'log/{path}/{log_name}'):
        open(f'log/{path}/{log_name}', 'w').close()
    file_handler: logging.FileHandler = logging.FileHandler(f'log/{path}/{log_name}')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler: logging.StreamHandler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
