import json
from argparse import Namespace
from pathlib import Path

from util import logging
from util.math import set_random_seed


class Experiment:
    __OUTPUT_PATH = 'out/certification'

    def __init__(self, settings: Namespace):
        self.settings = settings
        self.experiment_directory = self.__setup_experiment_directory()
        self.logger = logging.create_logger(self.experiment_directory)
        self.logger.info(settings)
        self.checkpoints = self.load_checkpoints()
        set_random_seed(settings.seed)

    def __setup_experiment_directory(self) -> Path:
        parent_dir = Path(Experiment.__OUTPUT_PATH)
        experiment_name = self.settings.experiment
        experiment_dir = parent_dir / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        return experiment_dir

    def load_checkpoints(self) -> dict:
        checkpoint_file = self.experiment_directory / 'certification_checkpoints.json'
        if checkpoint_file.is_file():
            with checkpoint_file.open('r') as json_file:
                return json.load(json_file)
        else:
            return {}

    def store_checkpoints(self, checkpoints: dict):
        checkpoint_file = self.experiment_directory / 'certification_checkpoints.json'
        with checkpoint_file.open('w') as json_file:
            json.dump(checkpoints, json_file, indent=4, sort_keys=True)
