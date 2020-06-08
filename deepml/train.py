import os
import csv

import numpy as np
import torch
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

from deepml import utils
from deepml.transforms import ImageNetInverseTransform
from deepml.constants import RUN_DIR_NAME


class Learner:

    def __init__(self, model, optimizer, model_save_path, model_file_name='latest_model.pt',
                 load_saved_model=False, load_optimizer_state=False, use_gpu=False):
        self.__model = model
        self.__optimizer = optimizer
        self.model_save_path = model_save_path
        self.epochs_completed = 0
        self.best_val_loss = np.inf
        self.lr_rates = {}
        os.makedirs(self.model_save_path, exist_ok=True)
        self.writer = SummaryWriter(os.path.join(self.model_save_path, RUN_DIR_NAME + utils.get_datetime()))
        self.device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

        if load_saved_model and model_file_name is not None:
            self.load_saved_model(os.path.join(self.model_save_path, model_file_name),
                                  load_optimizer_state)

    def load_saved_model(self, model_path, load_optimizer_state=False):
        if os.path.exists(model_path):
            print('Loading Saved Model Weights.')
            state_dict = torch.load(model_path)
            self.__model.load_state_dict(state_dict['model'])

            if load_optimizer_state and 'optimizer' in state_dict and state_dict['optimizer'] == \
                    self.__optimizer.__class__.__name__:
                self.__optimizer.load_state_dict(state_dict['optimizer_state'])

            if 'epoch' in state_dict:
                self.epochs_completed = state_dict['epoch']

            if 'metrics' in state_dict and 'val_loss' in state_dict['metrics']:
                self.best_val_loss = state_dict['metrics']['val_loss']
        else:
            print(f'{model_path} does not exist.')

    def save(self, model_file_name, save_optimizer_state=False, epoch=None, train_loss=None, val_loss=None):
        # Convert model into cpu before saving the model state
        cpu_model = self.__model.to("cpu")
        save_dict = {'model': cpu_model.state_dict()}

        if save_optimizer_state:
            save_dict['optimizer'] = self.__optimizer.__class__.__name__
            save_dict['optimizer_state'] = self.__optimizer.state_dict()

        if type(epoch) == int:
            save_dict['epoch'] = epoch

        if type(train_loss) == float and type(val_loss) == float:
            save_dict['metrics'] = {'train_loss': train_loss, 'val_loss': val_loss}

        torch.save(save_dict, os.path.join(self.model_save_path, model_file_name))

    def validate(self, criterion, loader, show_progress=True):
        if loader is None:
            raise Exception('Loader cannot be None.')

        self.__model.eval()
        running_loss = 0

        if show_progress:
            bar = tqdm(total=len(loader), desc="{:12s}".format('Validation'))

        with torch.no_grad():
            for batch_index, (X, y) in enumerate(loader):
                X = X.to(self.device)
                y = y.to(self.device)
                output = self.__model(X)
                loss = criterion(output, y)
                running_loss = running_loss + ((loss.item() - running_loss) / (batch_index + 1))

                if show_progress:
                    bar.update(1)
                    bar.set_postfix({'Val Loss': '{:.6f}'.format(running_loss)})

        return running_loss

    def fit(self, criterion, train_loader, val_loader=None, epochs=10, steps_per_epoch=None,
            save_model_after_every_epoch=5, lr_scheduler=None,
            viz_inverse_transform=ImageNetInverseTransform(), show_progress=True):
        """
        Starts training the model on specified train loader

        Parameters
        ----------
        criterion: loss function to optimize

        train_loader: The torch.utils.data.DataLoader for model to train on.

        val_loader: The torch.utils.data.DataLoader for model to validate on.
        Default is None.

        epochs: int The number of epochs to train. Default is 10

        steps_per_epoch: Should be around len(train_loader),
        so that every example in the dataset gets covered in each epoch.

        save_model_after_every_epoch: To save the model after every number of completed epochs
        Default is 5.

        lr_scheduler: the learning rate scheduler, default is None.

        viz_inverse_transform: It denotes reverse transformations of image normalization so that images
        can be displayed on tensor board. Default is deepml.transforms.ImageNetInverseTransform() which is
        an inverse of ImageNet normalization.

        show_progress: Show progress during training and validation. Default is True
        """
        if steps_per_epoch is None:
            steps_per_epoch = len(train_loader)

        self.__model = self.__model.to(self.device)
        criterion = criterion.to(self.device)

        # Write graph to tensorboard
        if torch.cuda.is_available():
            self.writer.add_graph(self.__model, next(iter(train_loader))[0].cuda())

        epochs = self.epochs_completed + epochs
        for epoch in range(self.epochs_completed, epochs):

            print('Epoch {}/{}:'.format(epoch + 1, epochs))

            # Training mode
            self.__model.train()

            # Iterate over batches
            step = 0
            lr_step_done = False
            running_train_loss = 0

            if show_progress:
                bar = tqdm(total=steps_per_epoch, desc="{:12s}".format('Training'))
            else:
                print('Training...')

            for batch_index, (X, y) in enumerate(train_loader):
                X = X.to(self.device)
                y = y.to(self.device)

                # zero the parameter gradients
                self.__optimizer.zero_grad()

                outputs = self.__model(X)
                loss = criterion(outputs, y)
                loss.backward()
                self.__optimizer.step()

                if isinstance(lr_scheduler, torch.optim.lr_scheduler.CyclicLR):
                    lr_scheduler.step()
                    lr_step_done = True

                step = step + 1
                running_train_loss = running_train_loss + ((loss.item() - running_train_loss) / step)

                if show_progress:
                    bar.update(1)
                    bar.set_postfix({'Train loss': '{:.6f}'.format(running_train_loss)})

                if step % steps_per_epoch == 0:
                    self.writer.add_images('Images/Train/Input/', viz_inverse_transform(X), epoch + 1)
                    if y.ndim > 1:
                        self.writer.add_images('Images/Train/Target', y, epoch + 1)
                        self.writer.add_images('Images/Train/Output', utils.binarize(outputs), epoch + 1)
                    break

            stats_info = 'Epoch: {}/{}\tTrain Loss: {:.6f}'.format(epoch + 1, epochs, running_train_loss)
            self.writer.add_scalar('Loss/Train', running_train_loss, epoch + 1)

            val_loss = np.inf
            if val_loader is not None:
                val_loss = self.validate(criterion, val_loader, show_progress)
                stats_info = stats_info + "\tVal Loss: {:.6f}".format(val_loss)
                self.writer.add_scalar('Loss/Val', val_loss, epoch + 1)

                # Log lass batch of val images to viz
                if viz_inverse_transform is not None and torch.cuda.is_available():
                    self.__model.eval()
                    with torch.no_grad():
                        X, y = next(iter(val_loader))
                        X, y = X.cuda(), y.cuda()
                        outputs = self.__model(X)
                        if y.ndim > 1:
                            self.writer.add_images('Images/Val/Input/', viz_inverse_transform(X), epoch + 1)
                            self.writer.add_images('Images/Val/Target', y)
                            self.writer.add_images('Images/Val/Output', utils.binarize(outputs), epoch + 1)

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
                              train_loss=running_train_loss,
                              val_loss=val_loss)

            if lr_scheduler is not None and not lr_step_done:
                lr_scheduler.step()
                lr_step_done = True

            # print training/validation statistics
            print(stats_info)
            self.writer.flush()
            if epoch % save_model_after_every_epoch == 0:
                model_file_name = "model_epoch_{}.pt".format(epoch)
                self.save(model_file_name, save_optimizer_state=True, epoch=epoch,
                          train_loss=running_train_loss, val_loss=val_loss)

        # Save latest model at the end
        self.save("latest_model.pt", save_optimizer_state=True, epoch=epoch, train_loss=running_train_loss,
                  val_loss=self.best_val_loss)

    def predict(self, loader):
        self.__model.eval()
        predictions = []
        targets = []
        with torch.no_grad():
            for X, y in tqdm(loader, total=len(loader), desc="{:12s}".format('Prediction')):
                X = X.to(self.device)
                pred = self.__model(X).cpu()
                predictions.append(pred)
                targets.append(y)

        predictions = torch.cat(predictions)
        targets = torch.cat(targets) if type(targets[0]) == torch.Tensor else np.hstack(targets).tolist()

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
                print('Iteration:', iteration + 1)
                for X, y in tqdm(loader, total=len(loader), desc='Feature Extraction'):
                    X = X.to(self.device)
                    feature_set = self.__model(X).cpu().numpy()

                    if target_known:
                        y = y.numpy().reshape(-1,1)
                        feature_set = np.hstack([y, feature_set])

                    csv_writer.writerows(feature_set)
                    fp.flush()
        fp.close()
