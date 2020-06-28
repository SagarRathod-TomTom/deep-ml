import os
import csv
from collections import OrderedDict

import numpy as np
import torch
import fastprogress
from fastprogress.fastprogress import master_bar, progress_bar

from torch.utils.tensorboard import SummaryWriter

from .predict import ImageRegressionPredictor
from .predict import ImageClassificationPredictor
from .predict import SemanticSegmentationPredictor

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
        self.device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

        self.__predictor = None
        self.__classes = classes
        self.__metrics_dict = OrderedDict({'loss': 0})

        if load_saved_model and model_file_name is not None:
            self.load_saved_model(os.path.join(self.model_save_path, model_file_name),
                                  load_optimizer_state)

        self.set_optimizer(self.__optimizer)

    def set_optimizer(self, optimizer):
        if optimizer is None:
            raise ValueError('Optimizer cannot be None.')

        self.__model = self.__model.to(self.device)
        self.__optimizer = optimizer.__class__(self.__model.parameters(),
                                               **optimizer.defaults)

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
        self.__model.to(self.device)

        return filepath

    def validate(self, criterion, loader, metrics=None,parent_bar=None):
        if loader is None:
            raise Exception('Loader cannot be None.')

        self.__model.eval()
        self.__metrics_dict['loss'] = 0
        self.__init_metrics(metrics)


        with torch.no_grad():

            if parent_bar is None:
                validation_bar = progress_bar(enumerate(loader), total=len(loader), parent=None)
                validation_bar.comment = f'Validation ...'
            else:
                validation_bar = progress_bar(enumerate(loader), total=len(loader), parent=parent_bar)
                parent_bar.child.comment = f'Validation ...'

            for batch_index, (X, y) in validation_bar:

                X = X.to(self.device)
                y = y.to(self.device)
                outputs = self.__model(X)

                if outputs.shape[1] == 1:
                    y = y.view_as(outputs)

                loss = criterion(outputs, y)

                self.__metrics_dict['loss'] = self.__metrics_dict['loss'] + ((loss.item() - self.__metrics_dict['loss'])
                                                                             / (batch_index + 1))

                self.__update_metrics(outputs, y, metrics, batch_index + 1)

        return self.__metrics_dict

    def __infer_predictor(self, x):

        self.__model.eval()
        with torch.no_grad():
            prediction = self.__model(x.to(self.device)).cpu()

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

    def __init_metrics(self, metrics):

        if metrics is None:
            return

        for metric in metrics:
            self.__metrics_dict[metric.__class__.__name__] = 0

    def __update_metrics(self, outputs, targets, metrics, step):

        if metrics is None:
            return

        # Update metrics
        outputs = outputs.to(self.device)
        targets = targets.to(self.device)

        for metric_obj in metrics:
            name = metric_obj.__class__.__name__
            self.__metrics_dict[name] = self.__metrics_dict[name] + \
                                        ((metric_obj(outputs, targets).item() - self.__metrics_dict[name]) / step)

    def __write_metrics_to_tensorboard(self, tag, global_step):
        for name, value in self.__metrics_dict.items():
            self.writer.add_scalar(f'{tag}/{name}', value, global_step)

    def __write_lr_to_tensorboard(self, global_step):
        # Write lr to tensor-board
        param_group = self.__optimizer.param_groups[0]
        self.writer.add_scalar('learning_rate', param_group['lr'], global_step)

    def fit(self, criterion, train_loader, val_loader=None, epochs=10, steps_per_epoch=None,
            save_model_after_every_epoch=5, lr_scheduler=None, image_inverse_transform=None,
            metrics=None, tboard_img_size=224):

        """
        Starts training the model on specified train loader

        Parameters
        ----------
        :param criterion: loss function to optimize

        :param train_loader: The torch.utils.data.DataLoader for model to train on.

        :param val_loader: The torch.utils.data.DataLoader for model to validate on.
        Default is None.

        :param epochs: int The number of epochs to train. Default is 10

        :param steps_per_epoch: Should be around len(train_loader),
        so that every example in the dataset gets covered in each epoch.

        :param save_model_after_every_epoch: To save the model after every number of completed epochs
        Default is 5.

        :param lr_scheduler: the learning rate scheduler, default is None.

        :param image_inverse_transform: It denotes reverse transformations of image normalization so that images

        can be displayed on tensor board. Default is deepml.transforms.ImageNetInverseTransform() which is
        an inverse of ImageNet normalization.

        :param metrics: list of metrics to monitor. Must be subclass of torch.nn.Module and implements
                        forward function

        :param tboard_img_size:  image size to use for writing images to tensorboard
        """
        if steps_per_epoch is None:
            steps_per_epoch = len(train_loader)

        self.__model = self.__model.to(self.device)
        criterion = criterion.to(self.device)

        if self.__predictor is None:
            x, _ = train_loader.dataset[0]
            # Add batch dimension
            x = x.unsqueeze(dim=0)
            self.__predictor = self.__infer_predictor(x)

        # Write graph to tensorboard
        if torch.cuda.is_available():
            self.writer.add_graph(self.__model, next(iter(train_loader))[0].cuda())

        # Check valid metrics types
        if metrics:
            for metric in metrics:
                if not (isinstance(metric, torch.nn.Module) and hasattr(metric, 'forward')):
                    raise TypeError(f'{metric.__class__} is not supported')

        train_loss = 0
        epoch = 0
        epochs = self.epochs_completed + epochs
                          
        epoch_bar = master_bar(range(self.epochs_completed, epochs))

        first_time = True


        for epoch in epoch_bar:

            epoch_bar.main_bar.comment = 'Epoch {}/{}:'.format(epoch + 1, epochs)

            # Training mode
            self.__model.train()

            # Iterate over batches
            step = 0
            lr_step_done = False

            # init all metrics with zeros
            self.__metrics_dict['loss'] = 0
            self.__init_metrics(metrics)


            # Write current lr to tensor-board
            self.__write_lr_to_tensorboard(epoch + 1)

            for batch_index, (X, y) in progress_bar(enumerate(train_loader),total=len(train_loader),parent=epoch_bar):

                epoch_bar.child.comment = f'Training ...'

                X = X.to(self.device)

                # zero the parameter gradients
                self.__optimizer.zero_grad()

                outputs = self.__model(X)

                if outputs.shape[1] == 1:
                    y = y.view_as(outputs)

                y = y.to(self.device)
                loss = criterion(outputs, y)
                loss.backward()

                self.__optimizer.step()

                if isinstance(lr_scheduler, torch.optim.lr_scheduler.CyclicLR):
                    lr_scheduler.step()
                    lr_step_done = True

                step = step + 1
                self.__metrics_dict['loss'] = self.__metrics_dict['loss'] + ((loss.item() - self.__metrics_dict['loss'])
                                                                             / step)
                # Update metrics
                self.__update_metrics(outputs, y, metrics, step)


            X, y = utils.get_random_samples_batch_from_loader(train_loader)
            X, y = X.to(self.device), y.to(self.device)
            self.__predictor.write_prediction_to_tensorboard('Train', (X, y),
                                                             self.writer, image_inverse_transform, epoch + 1)

            train_loss = self.__metrics_dict['loss']
            stats_info = 'Epoch: {}/{}\tTrain Loss: {:.6f}'.format(epoch + 1, epochs, self.__metrics_dict['loss'])
            self.epochs_completed = self.epochs_completed + 1

            self.__write_metrics_to_tensorboard('Train', epoch + 1)

            val_loss = np.inf
            if val_loader is not None:
                self.validate(criterion, val_loader, metrics,parent_bar=epoch_bar)
                val_loss = self.__metrics_dict['loss']
                stats_info = stats_info + "\tVal Loss: {:.6f}".format(self.__metrics_dict['loss'])
                self.__write_metrics_to_tensorboard('Val', epoch + 1)

                # write random val images to tensorboard
                X, y = utils.get_random_samples_batch_from_loader(val_loader)
                X, y = X.to(self.device), y.to(self.device)
                self.__predictor.write_prediction_to_tensorboard('Val', (X, y),
                                                                 self.writer, image_inverse_transform, epoch + 1)

                if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step(val_loss)
                    lr_step_done = True

                # Save best validation model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    print('Saving best validation model.')
                    self.save('best_val_model.pt',
                              save_optimizer_state=True,
                              epoch=epoch,
                              train_loss=train_loss,
                              val_loss=val_loss)

            if lr_scheduler is not None and not lr_step_done:
                lr_scheduler.step()
                lr_step_done = True

            # print training/validation statistics

            self.writer.flush()
            if epoch % save_model_after_every_epoch == 0:
                model_file_name = "model_epoch_{}.pt".format(epoch)
                self.save(model_file_name, save_optimizer_state=True, epoch=epoch,
                          train_loss=train_loss, val_loss=val_loss)

            if first_time:
                name = ["epoch", "train_loss", "valid_loss"]
                epoch_bar.write(name, table=True)
                first_time = False

            epoch_bar.write([epoch + 1, train_loss, val_loss], table=True)

        # Save latest model at the end
        self.save("latest_model.pt", save_optimizer_state=True, epoch=epoch, train_loss=train_loss,
                  val_loss=self.best_val_loss)

    def predict(self, loader):
        predictions, targets = self.__predictor.predict(loader, use_gpu=str(self.device) == "cuda:0")
        return predictions, targets

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
                feature_pbar = progress_bar(loader, total=len(loader), parent=None)
                feature_pbar.comment = f"Feature Extraction - Iteration: {iteration + 1}"
                for X, y in feature_pbar:
                    X = X.to(self.device)
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
