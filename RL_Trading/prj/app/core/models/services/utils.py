import logging
import sys


def get_logger(level: int) -> logging.Logger:
    formatter = "%(asctime)s [ %(threadName)s ] [ %(processName)s ] [ %(levelname)s ] [ %(funcName)s - line %(lineno)d ] - %(message)s"
    log_formatter = logging.Formatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(level)
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.addHandler(console_handler)
    return logger
