import os
from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
from torch.utils.tensorboard import SummaryWriter

from deepml import utils


class MLExperimentLogger(ABC):

    def __init__(self):
        super(MLExperimentLogger, self).__init__()

    @abstractmethod
    def log_params(self, **kwargs):
        pass

    @abstractmethod
    def log_metric(self, tag: str, value: Any, step: int):
        pass

    @abstractmethod
    def log_artifact(self, tag: str, value: Any, step: int, artifact_path: Optional[str] = None):
        pass


class TensorboardLogger(MLExperimentLogger):

    def __init__(self, model_dir):
        super().__init__()
        self.__model_dir = model_dir
        self.writer = SummaryWriter(os.path.join(self.__model_dir,
                                                 utils.find_new_run_dir_name(self.__model_dir)))

    def log_params(self, **kwargs):
        if 'task' in kwargs and 'loader' in kwargs:
            self.__write_graph_to_tensorboard(kwargs['task'], kwargs['loader'])
            self.writer.flush()

    def log_metric(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)
        self.writer.flush()

    def log_artifact(self, tag: str, value: Any, step: int, artifact_path: Optional[str] = None):

        if isinstance(value, torch.Tensor):
            self.writer.add_images(tag, torch.stack(value), step)
            self.writer.flush()

    def __write_graph_to_tensorboard(self, task, loader: torch.utils.data.DataLoader):

        if not loader:
            # Write graph to tensorboard
            temp_x = None
            for X, _ in loader:
                temp_x = X
                break

            temp_x = task.models_input_to_device(temp_x)

            with torch.no_grad():
                task.model.eval()
                try:
                    self.writer.add_graph(task.model, temp_x)
                except Exception as e:
                    print("Warning: Failed to write graph to tensorboard.", e)


class MLFlowLogger(MLExperimentLogger):

    try:
        import mlflow
    except ImportError as e:
        pass

    def __init__(self, experiment_name: str = "Default", tracking_uri: str = None):
        super().__init__()
        self.mlflow.set_experiment(experiment_name)

        if tracking_uri:
            self.mlflow.set_tracking_uri(tracking_uri)

    def log_params(self, **kwargs):
        self.mlflow.log_params(kwargs)

    def log_metric(self, tag: str, value: Any, step: int):
        self.mlflow.log_metric(tag, value, step)

    def log_artifact(self, tag: str, value: Any, step: int, artifact_path: Optional[str] = None):

        if isinstance(value, dict):
            self.mlflow.pytorch.log_model(value, tag)


class WandbLogger(MLExperimentLogger):

    try:
        import wandb
    except ImportError as e:
        pass

    def __init__(self, **kwargs: dict):
        super().__init__()
        if kwargs:
            self.wandb.init(*kwargs)

    def log_params(self, **kwargs):
        self.wandb.config.update(kwargs)

    def log_metric(self, tag: str, value: Any, step: int):
        self.wandb.log({tag: value})

    def log_artifact(self, tag: str, value: Any, step: int, artifact_path: Optional[str] = None):
        if artifact_path and os.path.exists(artifact_path):
            artifact = self.wandb.Artifact(name=tag, type='model_weights')
            artifact.add_file(local_path=artifact_path, name=tag)
            self.wandb.log_artifact(artifact)

        if isinstance(value, torch.Tensor) and value.ndim == 4:
            # TODO: log image
            pass

