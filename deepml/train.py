import os
import csv
from collections import OrderedDict

import numpy as np
import torch
from tqdm.auto import tqdm

from torch.utils.tensorboard import SummaryWriter

from deepml.predict import Predictor
from deepml.predict import ImageRegressionPredictor
from deepml.predict import ImageClassificationPredictor
from deepml.predict import SemanticSegmentationPredictor

from deepml import utils


class Learner:

    def __init__(self, model, optimizer, model_save_path, model_file_name='latest_model.pt',
                 load_saved_model=False, load_optimizer_state=True, use_gpu=False, classes=None):

        if model is None:
            raise ValueError('Model cannot be None.')

        if optimizer is None:
            raise ValueError('Optimizer cannot be None.')

        self.__model = model
        self.__optimizer = optimizer
        self.model_save_path = model_save_path
        self.epochs_completed = 0
        self.best_val_loss = np.inf

        os.makedirs(self.model_save_path, exist_ok=True)
        self.writer = SummaryWriter(os.path.join(self.model_save_path,
                                                 utils.find_new_run_dir_name(self.model_save_path)))

        self.__predictor = None
        self.__classes = classes
        self.__metrics_dict = OrderedDict({'loss': 0})

        if load_saved_model and model_file_name is not None:
            self.load_saved_model(os.path.join(self.model_save_path, model_file_name),
                                  load_optimizer_state)
        self.__device = "cpu"
        self.set_device("cuda" if use_gpu else "cpu")

    def set_optimizer(self, optimizer):

        assert isinstance(optimizer, torch.optim.Optimizer)
        self.__optimizer = optimizer

    def set_device(self, device):
        assert isinstance(device, str) and device.lower() in ['cpu', 'cuda']
        device = device.lower()
        self.__device = device if device == "cuda" and torch.cuda.is_available() else "cpu"

        self.__model.to(self.__device)

        for optim_state_values_dict in self.__optimizer.state.values():
            for key in optim_state_values_dict:
                if type(optim_state_values_dict[key]) == torch.Tensor:
                    optim_state_values_dict[key] = optim_state_values_dict[key].to(self.__device)

    def load_saved_model(self, model_path, load_optimizer_state=False):
        if os.path.exists(model_path):
            print('Loading Saved Model Weights.')
            state_dict = torch.load(model_path)
            self.__model.load_state_dict(state_dict['model'])

            if load_optimizer_state and 'optimizer' in state_dict:
                if state_dict['optimizer'] == self.__optimizer.__class__.__name__:
                    self.__optimizer.load_state_dict(state_dict['optimizer_state'])
                else:
                    print(f"Skipping load optimizer state because {self.__optimizer.__class__.__name__}"
                          f" != {state_dict['optimizer']}")

            if 'epoch' in state_dict:
                self.epochs_completed = state_dict['epoch']

            if 'metrics' in state_dict and 'val_loss' in state_dict['metrics']:
                self.best_val_loss = state_dict['metrics']['val_loss']

            if 'classes' in state_dict:
                self.__classes = state_dict['classes']

            if 'predictor' in state_dict:
                # instantiate predictor class
                self.__predictor = state_dict['predictor'](model=self.__model, classes=self.__classes)

        else:
            print(f'{model_path} does not exist.')

    def save(self, model_file_name, save_optimizer_state=False, epoch=None, train_loss=None, val_loss=None):
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

        if self.__predictor is not None:
            save_dict['predictor'] = self.__predictor.__class__

        if self.__classes is not None:
            save_dict['classes'] = self.__classes

        filepath = os.path.join(self.model_save_path, model_file_name)
        torch.save(save_dict, filepath)

        self.__model.to(self.__device)
        return filepath

    def validate(self, criterion, loader, metrics=None):
        if loader is None:
            raise Exception('Loader cannot be None.')

        self.__model.eval()
        self.__metrics_dict['loss'] = 0
        self.__init_metrics(metrics)

        bar = tqdm(total=len(loader), desc="{:12s}".format('Validation'))

        with torch.no_grad():
            for batch_index, (X, y) in enumerate(loader):

                X = X.to(self.__device)
                y = y.to(self.__device)
                outputs = self.__model(X)

                if outputs.shape[1] == 1:
                    y = y.view_as(outputs)

                loss = criterion(outputs, y)

                self.__metrics_dict['loss'] = self.__metrics_dict['loss'] + ((loss.item() - self.__metrics_dict['loss'])
                                                                             / (batch_index + 1))
                self.__update_metrics(outputs, y, metrics, batch_index + 1)

                bar.update(1)
                bar.set_postfix({name: f'{round(value, 2)}' for name, value in self.__metrics_dict.items()})

        return self.__metrics_dict

    def __infer_predictor(self, x):

        self.__model.eval()
        with torch.no_grad():
            prediction = self.__model(x.to(self.__device)).cpu()

            if self.__classes is None:
                try:
                    prediction = prediction.item()
                    return ImageRegressionPredictor(self.__model)
                except ValueError:
                    pass

            if prediction.ndim == 2:
                return ImageClassificationPredictor(self.__model, classes=self.__classes)
            else:
                return SemanticSegmentationPredictor(self.__model, classes=self.__classes)

    def set_predictor(self, predictor):
        assert isinstance(predictor, Predictor)
        self.__predictor = predictor

    def __init_metrics(self, metrics):
        if metrics is None:
            return
        unique_metrics = set()
        for metric_name, _ in metrics:
            if metric_name in unique_metrics:
                raise ValueError("Metrics names should be unique")
            unique_metrics.add(metric_name)
            self.__metrics_dict[metric_name] = 0

    def __update_metrics(self, outputs, targets, metrics, step):

        if metrics is None:
            return

        # Update metrics
        outputs = outputs.to(self.__device)
        targets = targets.to(self.__device)

        for metric_name, metric_instance in metrics:
            self.__metrics_dict[metric_name] = self.__metrics_dict[metric_name] + \
                                               ((metric_instance(outputs, targets).item() - self.__metrics_dict[
                                                   metric_name]) / step)

    def __write_metrics_to_tensorboard(self, tag, global_step):
        for name, value in self.__metrics_dict.items():
            self.writer.add_scalar(f'{tag}/{name}', value, global_step)

    def __write_lr_to_tensorboard(self, global_step):
        # Write lr to tensor-board
        if len(self.__optimizer.param_groups) == 1:
            param_group = self.__optimizer.param_groups[0]
            self.writer.add_scalar('learning_rate', param_group['lr'], global_step)
        else:
            for index, param_group in enumerate(self.__optimizer.param_groups):
                self.writer.add_scalar(f'learning_rate/param_group_{index}', param_group['lr'],
                                       global_step)

    def fit(self, criterion, train_loader, val_loader=None, epochs=10, steps_per_epoch=None,
            save_model_after_every_epoch=5, lr_scheduler=None, lr_scheduler_step_policy='epoch',
            image_inverse_transform=None, metrics=None, tboard_img_size=224):

        """
        Trains the model on specified train loader for specified number of epochs.

        Parameters
        ----------
        :param criterion: loss function to optimize

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

        :param metrics: list of tuples ('metric_name', metric instance) to monitor.
                        Metric name is used as label for logging metric value to console and tensorboard.
                        Metric instance must be subclass of torch.nn.Module, implements forward function and
                        returns calculated value.

        :param tboard_img_size:  image size to use for writing images to tensorboard
        """
        if steps_per_epoch is None:
            steps_per_epoch = len(train_loader)

        self.__model.to(self.__device)
        criterion = criterion.to(self.__device)

        if self.__predictor is None:
            x, _ = train_loader.dataset[0]
            # Add batch dimension
            x = x.unsqueeze(dim=0)
            self.__predictor = self.__infer_predictor(x)

        # Write graph to tensorboard
        temp_x, _ = train_loader.dataset[0]
        self.writer.add_graph(self.__model, temp_x.unsqueeze(dim=0).to(self.__device))

        # Check valid metrics types
        if metrics:
            for metric_name, metric_instance in metrics:
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

            # Write current lr to tensor-board
            self.__write_lr_to_tensorboard(epoch + 1)

            bar = tqdm(total=steps_per_epoch, desc="{:12s}".format('Training'))
            for batch_index, (X, y) in enumerate(train_loader):

                X = X.to(self.__device)

                # zero the parameter gradients
                self.__optimizer.zero_grad()

                outputs = self.__model(X)

                if outputs.shape[1] == 1:
                    y = y.view_as(outputs)

                y = y.to(self.__device)
                loss = criterion(outputs, y)
                loss.backward()

                self.__optimizer.step()

                if lr_scheduler is not None and lr_scheduler_step_policy == "batch":
                    lr_scheduler.step()

                step = step + 1
                self.__metrics_dict['loss'] = self.__metrics_dict['loss'] + ((loss.item() - self.__metrics_dict['loss'])
                                                                             / step)
                # Update metrics
                self.__update_metrics(outputs, y, metrics, step)
                bar.update(1)
                bar.set_postfix({name: f'{round(value, 2)}' for name, value in self.__metrics_dict.items()})

            self.epochs_completed = self.epochs_completed + 1

            # Write some sample training images to tensorboard
            X, y = utils.get_random_samples_batch_from_loader(train_loader)
            X, y = X.to(self.__device), y.to(self.__device)
            self.__predictor.write_prediction_to_tensorboard('Train', (X, y),
                                                             self.writer, image_inverse_transform,
                                                             self.epochs_completed, img_size=tboard_img_size)

            train_loss = self.__metrics_dict['loss']
            self.__write_metrics_to_tensorboard('Train', self.epochs_completed)

            val_loss = np.inf
            if val_loader is not None:
                self.validate(criterion, val_loader, metrics)
                val_loss = self.__metrics_dict['loss']
                self.__write_metrics_to_tensorboard('Val', self.epochs_completed)

                # write random val images to tensorboard
                X, y = utils.get_random_samples_batch_from_loader(val_loader)
                X, y = X.to(self.__device), y.to(self.__device)
                self.__predictor.write_prediction_to_tensorboard('Val', (X, y),
                                                                 self.writer, image_inverse_transform,
                                                                 self.epochs_completed,
                                                                 img_size=tboard_img_size)

                # Save best validation model
                if val_loss < self.best_val_loss:
                    print("Saving best validation model.")
                    self.best_val_loss = val_loss
                    self.save('best_val_model.pt',
                              save_optimizer_state=True,
                              epoch=self.epochs_completed,
                              train_loss=train_loss,
                              val_loss=val_loss)

            if lr_scheduler is not None and lr_scheduler_step_policy == "epoch":
                if val_loader is not None and isinstance(lr_scheduler,
                                                         torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step(val_loss)
                else:
                    lr_scheduler.step()

            self.writer.flush()
            if epoch % save_model_after_every_epoch == 0:
                model_file_name = "model_epoch_{}.pt".format(epoch)
                self.save(model_file_name, save_optimizer_state=True, epoch=self.epochs_completed,
                          train_loss=train_loss, val_loss=val_loss)

        # Save latest model at the end
        self.save("latest_model.pt", save_optimizer_state=True, epoch=self.epochs_completed,
                  train_loss=train_loss, val_loss=self.best_val_loss)

    def predict(self, loader):
        predictions, targets = self.__predictor.predict(loader, use_gpu=self.__device == "cuda")
        return predictions, targets

    def predict_class(self, loader):
        predicted_class, probability, targets = self.__predictor.predict_class(loader,
                                                                               use_gpu=self.__device == "cuda")
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
                for X, y in tqdm(loader, total=len(loader), desc='Feature Extraction'):
                    X = X.to(self.__device)
                    feature_set = self.__model(X).cpu().numpy()

                    if target_known:
                        y = y.numpy().reshape(-1, 1)
                        feature_set = np.hstack([y, feature_set])

                    csv_writer.writerows(feature_set)
                    fp.flush()
        fp.close()

    def show_predictions(self, loader, image_inverse_transform=None, samples=9, cols=3, figsize=(10, 10)):

        self.__predictor.show_predictions(loader, image_inverse_transform=image_inverse_transform,
                                          samples=samples, cols=cols, figsize=figsize)
