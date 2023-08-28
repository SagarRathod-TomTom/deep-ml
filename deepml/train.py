import os
import csv
from collections import OrderedDict, defaultdict
from typing import List, Tuple, Callable, Union, Dict

import numpy as np
import torch
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

import deepml.tasks
from deepml.tasks import Task
from deepml.tracking import MLExperimentLogger, TensorboardLogger


class Learner:

    def __init__(self, task: Task, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module,
                 load_optimizer_state: bool = False, logger: MLExperimentLogger = None):
        """
        Training class for learning a model weights using predictor and optimizer.

        :param task: Object of sub class deepml.tasks.Task
        :param optimizer: The optimizer from torch.optim
        :param criterion: The loss function
        :param load_optimizer_state: Weather to load optimizer state to resume model training. Default is False.
                                     If true, optimizer state is loaded with load_state_dict and history of epoch.
        """

        assert isinstance(task, Task)
        assert isinstance(optimizer, torch.optim.Optimizer)
        assert isinstance(criterion, torch.nn.Module)

        self.__predictor = task
        self.__model = self.__predictor.model
        self.__model_dir = self.__predictor.model_dir
        self.__model_file_name = self.__predictor.model_file_name
        self.__optimizer = optimizer
        self.__criterion = criterion
        self.epochs_completed = 0
        self.best_val_loss = np.inf
        self.history = defaultdict(list)
        self.logger = logger

        if self.logger is None:
            os.makedirs(self.__model_dir, exist_ok=True)
            self.logger = TensorboardLogger(self.__model_dir)

        self.__metrics_dict = OrderedDict({'loss': 0})

        if load_optimizer_state:
            self.__load_optimizer_state()

        self.__device = self.__predictor.device
        self.set_device(self.__device)

    def set_optimizer(self, optimizer: torch.optim.Optimizer):
        assert isinstance(optimizer, torch.optim.Optimizer)
        self.__optimizer = optimizer

    def set_criterion(self, criterion: torch.nn.Module):
        assert isinstance(criterion, torch.nn.Module)
        self.__criterion = criterion

    def set_device(self, device: str):
        assert isinstance(device, str) and device.lower() in ['cpu', 'cuda']
        device = device.lower()
        self.__device = device if device == "cuda" and torch.cuda.is_available() else "cpu"

        self.__model.to(self.__device)

        for optim_state_values_dict in self.__optimizer.state.values():
            for key in optim_state_values_dict:
                if type(optim_state_values_dict[key]) == torch.Tensor:
                    optim_state_values_dict[key] = optim_state_values_dict[key].to(self.__device)

    def __load_optimizer_state(self):
        model_path = os.path.join(self.__model_dir, self.__model_file_name)
        if os.path.exists(model_path):
            state_dict = (torch.load(model_path) if torch.cuda.is_available()
                          else torch.load(model_path, map_location=torch.device('cpu')))
            if 'optimizer' in state_dict:
                if state_dict['optimizer'] == self.__optimizer.__class__.__name__:
                    self.__optimizer.load_state_dict(state_dict['optimizer_state'])
                else:
                    print(f"Skipping load optimizer state because {self.__optimizer.__class__.__name__}"
                          f" != {state_dict['optimizer']}")

            if 'epoch' in state_dict:
                self.epochs_completed = state_dict['epoch']

            if 'metrics' in state_dict and 'val_loss' in state_dict['metrics']:
                self.best_val_loss = state_dict['metrics']['val_loss']
        else:
            print(f'{model_path} does not exist.')

    def save(self, tag: str, save_optimizer_state: bool = False, epoch: int = -1, train_loss: float = None,
             val_loss: float = None):
        # Convert model into cpu before saving the model state
        self.__model.to("cpu")
        save_dict = {'model': self.__model.state_dict()}

        if save_optimizer_state:
            save_dict['optimizer'] = self.__optimizer.__class__.__name__
            save_dict['optimizer_state'] = self.__optimizer.state_dict()

        if type(epoch) == int:
            save_dict['epoch'] = epoch

        if type(train_loss) == float and type(val_loss) == float:
            save_dict['metrics'] = {'train_loss': train_loss, 'val_loss': val_loss}

        save_dict['criterion'] = self.__criterion.__class__.__name__

        filepath = f"{os.path.join(self.__model_dir, tag)}.pt"
        torch.save(save_dict, filepath)

        self.logger.log_artifact(tag, self.__model, epoch, artifact_path=filepath)
        self.__model.to(self.__device)
        return filepath

    def validate(self, loader: torch.utils.data.DataLoader, criterion: torch.nn.Module,
                 metrics: Dict[str, torch.nn.Module] = None, non_blocking=False):
        if loader is None:
            raise Exception('Loader cannot be None.')

        self.__model.eval()
        self.__metrics_dict['loss'] = 0
        self.__init_metrics(metrics)

        bar = tqdm(total=len(loader), desc="{:12s}".format('Validation'))

        with torch.no_grad():
            for batch_index, (x, y) in enumerate(loader):

                outputs = self.__predictor.predict_batch(x, y, non_blocking)

                if isinstance(y, torch.Tensor):
                    y = y.to(self.__device)

                if isinstance(outputs, torch.Tensor) and outputs.ndim == 2 and outputs.shape[1] == 1:
                    y = y.view_as(outputs)

                loss = criterion(outputs, y)

                self.__metrics_dict['loss'] = self.__metrics_dict['loss'] + ((loss.item() - self.__metrics_dict['loss'])
                                                                             / (batch_index + 1))
                self.__update_metrics(outputs, y, metrics, batch_index + 1)

                bar.update(1)
                bar.set_postfix({name: f'{round(value, 4)}' for name, value in self.__metrics_dict.items()})

        return self.__metrics_dict

    def set_predictor(self, predictor: deepml.tasks.Task):
        assert isinstance(predictor, Task)
        self.__predictor = predictor

    def __init_metrics(self, metrics:  Dict[str, torch.nn.Module]):
        if metrics is None:
            return

        for metric_name, _ in metrics.items():
            if metric_name == "loss":
                raise ValueError("Metric name 'loss' is reserved of criterion")
            self.__metrics_dict[metric_name] = 0

    def __update_metrics(self, outputs: torch.Tensor, targets: torch.Tensor, metrics: Dict[str, torch.nn.Module],
                         step: int):

        if metrics is None:
            return

        with torch.no_grad():
            for metric_name, metric_instance in metrics.items():
                self.__metrics_dict[metric_name] = self.__metrics_dict[metric_name] + \
                                                   ((metric_instance(outputs, targets).item() - self.__metrics_dict[
                                                       metric_name]) / step)

    def __write_metrics_to_logger(self, tag: str, global_step: int):
        for name, value in self.__metrics_dict.items():
            self.logger.log_metric(f'{name}/{tag}', value, global_step)

    def __write_history(self, stage: str):
        for name, value in self.__metrics_dict.items():
            self.history[f"{stage}_{name}"].append(value)

    def __write_lr(self, global_step: int):
        # Write lr to tensor-board and history dict
        if len(self.__optimizer.param_groups) == 1:
            param_group = self.__optimizer.param_groups[0]
            self.logger.log_metric('learning_rate', param_group['lr'], global_step)
            self.history['learning_rate'].append(param_group['lr'])
        else:
            for index, param_group in enumerate(self.__optimizer.param_groups):
                self.logger.log_metric(f'learning_rate/param_group_{index}', param_group['lr'],
                                       global_step)
                self.history[f'learning_rate/param_group_{index}'].append(param_group['lr'])

    def fit(self, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader = None,
            epochs: int = 10, steps_per_epoch: int = None,
            save_model_after_every_epoch: int = 5, lr_scheduler=None, lr_scheduler_step_policy: str = 'epoch',
            metrics: Dict[str, torch.nn.Module] = None, image_inverse_transform: Callable = None,
            logger_img_size=Union[int, Tuple[int, int]],
            non_blocking=False):

        """
        Trains the model on specified train loader for specified number of epochs.

        Parameters
        ----------
        :param train_loader: The torch.utils.data.DataLoader for model to train on.

        :param val_loader: The torch.utils.data.DataLoader for model to validate on.
                           Default is None.

        :param epochs: int The number of epochs to train. Default is 10

        :param steps_per_epoch: Should be around len(train_loader), so that every example in the
                                dataset gets covered in each epoch.

        :param save_model_after_every_epoch: To save the model after every number of completed epochs
                                            Default is 5.

        :param lr_scheduler: the learning rate scheduler, default is None.

        :param lr_scheduler_step_policy: It is the time when lr_scheduler.step() would be called.
                                         Default is "epoch" policy.
                                         Use "batch" policy if you want lr_scheduler.step() to be
                                         called after each gradient step.

        :param image_inverse_transform: It denotes reverse transformations of image normalization so that images
                                        can be displayed on tensor board.
                                        Default is deepml.transforms.ImageNetInverseTransform() which is
                                        an inverse of ImageNet normalization.

        :param metrics: dictionary of metrics 'metric_name': metric instance to monitor.
                        Metric name is used as label for logging metric value to console and tensorboard.
                        Metric instance must be subclass of torch.nn.Module, which implements forward function and
                        returns calculated value.

        :param logger_img_size:  image size to use for writing images to tensorboard

        :param non_blocking:  weather to enable asynchronous cuda tensor transfer
        """
        if steps_per_epoch is None:
            steps_per_epoch = len(train_loader)

        assert steps_per_epoch <= len(train_loader), "Steps per epoch should not be greater than len(train_loader)"

        self.__model.to(self.__device)
        self.__criterion = self.__criterion.to(self.__device)

        # Log params
        self.logger.log_params(task=self.__predictor, loader=val_loader, epochs=epochs, criterion=self.__criterion,
                               lr_scheduler=lr_scheduler)

        # Check valid metrics types
        if metrics:
            for metric_name, metric_instance in metrics.items():
                if not (isinstance(metric_instance, torch.nn.Module) and hasattr(metric_instance, 'forward')):
                    raise TypeError(f'{metric_instance.__class__} is not supported')

        # Check valid policy for lr_scheduler
        if lr_scheduler is not None:
            lr_scheduler_step_policy = lr_scheduler_step_policy.lower()
            assert isinstance(lr_scheduler_step_policy, str) and lr_scheduler_step_policy in ['epoch', 'batch']

        # Replace all metrics during call to learner fit
        self.__metrics_dict = OrderedDict({'loss': 0})

        train_loss = 0
        epochs = self.epochs_completed + epochs
        for epoch in range(self.epochs_completed, epochs):
            print('Epoch {}/{}:'.format(epoch + 1, epochs))
            # Training mode
            self.__model.train()

            # Iterate over batches
            step = 0

            # init all metrics with zeros
            self.__metrics_dict['loss'] = 0
            self.__init_metrics(metrics)

            # Write current lr to logger
            self.__write_lr(epoch + 1)

            bar = tqdm(total=steps_per_epoch, desc="{:12s}".format('Training'))
            for batch_index, (x, y) in enumerate(train_loader):

                # zero the parameter gradients
                self.__optimizer.zero_grad()

                if isinstance(y, torch.Tensor):
                    y = y.to(self.__device)

                outputs = self.__predictor.predict_batch(x, y, non_blocking)

                if isinstance(outputs, torch.Tensor) and outputs.ndim == 2 and outputs.shape[1] == 1:
                    y = y.view_as(outputs)

                loss = self.__criterion(outputs, y)
                loss.backward()

                self.__optimizer.step()

                if lr_scheduler and lr_scheduler_step_policy == "batch":
                    lr_scheduler.step()

                step = step + 1
                self.__metrics_dict['loss'] = self.__metrics_dict['loss'] + ((loss.item() - self.__metrics_dict['loss'])
                                                                             / step)
                # Update metrics
                self.__update_metrics(outputs, y, metrics, step)
                bar.update(1)
                bar.set_postfix({name: f'{round(value, 4)}' for name, value in self.__metrics_dict.items()})

                if (batch_index + 1) % steps_per_epoch == 0:
                    break

            self.epochs_completed = self.epochs_completed + 1

            # Write some sample training images to logger
            self.__predictor.write_prediction_to_logger('train', train_loader,
                                                        self.logger, image_inverse_transform,
                                                        self.epochs_completed, img_size=logger_img_size)

            train_loss = self.__metrics_dict['loss']
            self.__write_metrics_to_logger('train', self.epochs_completed)
            self.__write_history('train')
            message = f"Training Loss: {train_loss:.4f} "
            val_loss = np.inf
            if val_loader is not None:
                self.validate(val_loader, self.__criterion, metrics)
                val_loss = self.__metrics_dict['loss']
                self.__write_metrics_to_logger('val', self.epochs_completed)
                self.__write_history('val')
                message = message + f"Validation Loss: {val_loss:.4f} "
                # write random val images to tensorboard
                self.__predictor.write_prediction_to_logger('val', val_loader,
                                                            self.logger, image_inverse_transform,
                                                            self.epochs_completed,
                                                            img_size=logger_img_size)
                # Save best validation model
                if val_loss < self.best_val_loss:
                    message = message + "[Saving best validation model]"
                    self.best_val_loss = val_loss
                    self.save('best_val_model',
                              save_optimizer_state=True,
                              epoch=self.epochs_completed,
                              train_loss=train_loss,
                              val_loss=val_loss)

            if lr_scheduler and lr_scheduler_step_policy == "epoch":
                if val_loader and isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step(val_loss)
                else:
                    lr_scheduler.step()

            print(message)
            if self.epochs_completed % save_model_after_every_epoch == 0:
                model_tag_name = "epoch_{}_model".format(self.epochs_completed)
                self.save(model_tag_name, save_optimizer_state=True, epoch=self.epochs_completed,
                          train_loss=train_loss, val_loss=val_loss)

        # Save latest model at the end
        self.save("latest_model", save_optimizer_state=True, epoch=self.epochs_completed,
                  train_loss=train_loss, val_loss=self.best_val_loss)

    def predict(self, loader):
        predictions, targets = self.__predictor.predict(loader)
        return predictions, targets

    def predict_class(self, loader):
        predicted_class, probability, targets = self.__predictor.predict_class(loader)
        return predicted_class, probability, targets

    def extract_features(self, loader, no_of_features, features_csv_file, iterations=1,
                         target_known=True):

        fp = open(features_csv_file, 'w')
        csv_writer = csv.writer(fp)

        # define feature columns
        cols = ["feat_{}".format(i) for i in range(0, no_of_features)]

        if target_known:
            cols = ["class"] + cols

        csv_writer.writerow(cols)
        fp.flush()

        self.__model.eval()
        with torch.no_grad():
            for iteration in range(iterations):
                print('Iteration:', iteration + 1)
                for x, y in tqdm(loader, total=len(loader), desc='Feature Extraction'):

                    feature_set = self.__predictor.predict_batch(x).cpu().numpy()

                    if target_known:
                        y = y.numpy().reshape(-1, 1)
                        feature_set = np.hstack([y, feature_set])

                    csv_writer.writerows(feature_set)
                    fp.flush()
        fp.close()

    def show_predictions(self, loader, image_inverse_transform=None, samples=9, cols=3, figsize=(10, 10),
                         target_known=True):

        self.__predictor.show_predictions(loader, image_inverse_transform=image_inverse_transform,
                                          samples=samples, cols=cols, figsize=figsize, target_known=target_known)
