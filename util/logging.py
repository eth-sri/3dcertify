import datetime
import logging
import sys
from pathlib import Path
from typing import Union


def create_logger(log_dir: Union[str, Path], log_name: str = "certification") -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(Path(log_dir) / f"{log_name}_{datetime.datetime.now().isoformat(timespec='seconds')}.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger
