import os
import csv

import numpy as np
import torch
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

from deepml import utils
from deepml.constants import RUN_DIR_NAME
from deepml.transforms import ImageNetInverseTransform


class Trainer:

    def __init__(self, model, optimizer, model_save_path, load_saved_model=False,
                 model_file_name='latest_model.pt'):
        self.model = model
        self.optimizer = optimizer
        self.model_save_path = model_save_path
        self.epochs_completed = 0
        self.best_val_loss = np.inf
        self.lr_rates = {}
        self.writer = SummaryWriter(os.path.join(model_save_path, RUN_DIR_NAME +
                                                 str(utils.find_current_run_number(model_save_path))))

        if load_saved_model:
            self.load_saved_model(model_file_name)

    def load_saved_model(self, model_file_name):
        print('Loading Saved Model Weights.')
        state_dict = torch.load(os.path.join(self.model_save_path, model_file_name))
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.epochs_completed = state_dict['epoch']
        self.best_val_loss = state_dict['metrics']['val_loss']

    def save_model_optim_state(self, model_file_name, epoch, train_loss, val_loss):
        torch.save({'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'metrics': {'train_loss': train_loss, 'val_loss': val_loss},
                    'epoch': epoch
                    },
                    os.path.join(self.model_save_path, model_file_name))

    def validate(self, criterion, loader, use_gpu=False, show_progress=True):
        if loader is None:
            raise Exception('Loader cannot be None.')

        device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)

        self.model.eval()
        running_loss = 0

        if show_progress:
            bar = tqdm(total=len(loader), desc="{:12s}".format('Validation'))

        with torch.no_grad():
            for batch_index, (X, y) in enumerate(loader):
                if use_gpu:
                    X = X.to(device)
                    y = y.to(device)
                output = self.model(X)
                loss = criterion(output, y)
                running_loss = running_loss + ((loss.item() - running_loss) / (batch_index + 1))

                if show_progress:
                    bar.update(1)
                    bar.set_postfix({'Val Loss': '{:.6f}'.format(running_loss)})

        return running_loss

    def fit(self, criterion, train_loader, val_loader=None, epochs=10, use_gpu=False,
              steps_per_epoch=None, save_model_afer_every_epoch=5, lr_scheduler=None,
            viz_reverse_transform=None, show_progress=True):

        """
        steps_per_epoch = should be around len(train_loader),
            so that every example in the dataset gets covered in each epoch.
        """
        device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)

        if steps_per_epoch is None:
            steps_per_epoch = len(train_loader)

        # Write graph to tensorboard
        if torch.cuda.is_available():
            self.writer.add_graph(self.model, next(iter(train_loader))[0].cuda())

        if viz_reverse_transform is None:
            viz_reverse_transform = ImageNetInverseTransform()

        epochs = self.epochs_completed + epochs
        for epoch in range(self.epochs_completed, epochs):

            print('Epoch {}/{}:'.format(epoch + 1, epochs))

            # Training mode
            self.model.train()

            # Iterate over batches
            step = 0
            lr_step_done = False
            running_train_loss = 0

            if show_progress:
                bar = tqdm(total=steps_per_epoch, desc="{:12s}".format('Training'))
            else:
                print('Training...')

            for batch_index, (X, y) in enumerate(train_loader):

                if use_gpu:
                    X = X.to(device)
                    y = y.to(device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                outputs = self.model(X)
                loss = criterion(outputs, y)
                loss.backward()
                self.optimizer.step()

                if isinstance(lr_scheduler, torch.optim.lr_scheduler.CyclicLR):
                    lr_scheduler.step()
                    lr_step_done = True

                step = step + 1
                running_train_loss = running_train_loss + ((loss.item() - running_train_loss) / step)

                if show_progress:
                    bar.update(1)
                    bar.set_postfix({'Train loss': '{:.6f}'.format(running_train_loss)})

                if step % steps_per_epoch == 0:
                    self.writer.add_images('Images/Train/Input/', viz_reverse_transform(X), epoch + 1)
                    self.writer.add_images('Images/Train/Target', y, epoch + 1)
                    self.writer.add_images('Images/Train/Output', utils.binarize(outputs), epoch + 1)
                    break

            stats_info = 'Epoch: {}/{}\tTrain Loss: {:.6f}'.format(epoch + 1, epochs, running_train_loss)
            self.writer.add_scalar('Loss/Train', running_train_loss, epoch + 1)

            val_loss = np.inf
            if val_loader is not None:
                val_loss = self.validate(criterion, val_loader, use_gpu, show_progress)
                stats_info = stats_info + "\tVal Loss: {:.6f}".format(val_loss)
                self.writer.add_scalar('Loss/Val', val_loss, epoch + 1)

                # Log lass batch of val images to viz
                if viz_reverse_transform is not None and torch.cuda.is_available():
                    self.model.eval()
                    with torch.no_grad():
                        X, y = next(iter(val_loader))
                        X, y = X.cuda(), y.cuda()
                        outputs = self.model(X)
                        self.writer.add_images('Images/Val/Input/', viz_reverse_transform(X), epoch + 1)
                        self.writer.add_images('Images/Val/Target', y)
                        self.writer.add_images('Images/Val/Output', utils.binarize(outputs), epoch + 1)

                if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step(val_loss)
                    lr_step_done = True

                # Save best validation model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    print('Saving best validation model.')
                    self.save_model_optim_state('best_val_model.pt', epoch, train_loss=running_train_loss,
                                                val_loss=val_loss)

            if lr_scheduler is not None and not lr_step_done:
                lr_scheduler.step()
                lr_step_done = True

            # print training/validation statistics
            print(stats_info)
            self.writer.flush()
            if epoch % save_model_afer_every_epoch == 0:
                model_file_name = "model_epoch_{}.pt".format(epoch)
                self.save_model_optim_state(model_file_name, epoch, train_loss=running_train_loss, val_loss=val_loss)

        self.writer.close()
        # Save latest model at the end
        self.save_model_optim_state("latest_model.pt", epoch, train_loss=running_train_loss,
                                    val_loss=self.best_val_loss)

    def predict_proba(self, loader, use_gpu=False):

        device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)

        self.model.eval()
        preds = []
        y_true = []
        with torch.no_grad():
            for X, y in tqdm(loader, total=len(loader), desc="{:12s}".format('Prediction')):
                if use_gpu:
                    X = X.to(device)
                pred = self.model(X).cpu()
                preds.append(pred)
                y_true.append(y)

        return torch.cat(preds), torch.cat(y_true)

    def extract_features(self, loader, no_of_features, features_csv_file, iterations=1,
                         target_known=True, use_gpu=False):

        device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)

        fp = open(features_csv_file, 'w')
        csv_writer = csv.writer(fp)

        # define feature columns
        cols = ["feat_{}".format(i) for i in range(0, no_of_features)]

        if target_known:
            cols = ["class"] + cols

        csv_writer.writerow(cols)
        fp.flush()

        self.model.eval()
        with torch.no_grad():
            for iteration in range(iterations):
                print('Iteration:', iteration + 1)
                for X, y in tqdm(loader, total=len(loader), desc='Feature Extraction'):
                    X = X.to(device)
                    feature_set = self.model(X).cpu().numpy()

                    if target_known:
                        y = y.numpy().reshape(-1,1)
                        feature_set = np.hstack([y, feature_set])

                    csv_writer.writerows(feature_set)
                    fp.flush()
        fp.close()
