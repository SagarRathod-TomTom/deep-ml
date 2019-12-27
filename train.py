import os
import numpy as np
import torch
from torch import optim
from tqdm import tqdm_notebook
import csv


class Trainer:
    def __init__(self, model, optimizer, model_save_path, load_saved_model=False,
                 model_file_name='latest_model.pt'):
        self.model = model
        self.optimizer = optimizer
        self.model_save_path = model_save_path
        self.epochs_completed = 0
        self.best_val_loss = np.inf
        self.lr_rates = {}

        if load_saved_model:
            print('Loading Saved Model Weights.')
            self.load_saved_model(model_file_name)

    def load_saved_model(self, model_file_name):
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

    def evaluate(self, criterion, loader, use_gpu=False):
        if loader is None:
            raise Exception('Loader cannot be None.')

        if use_gpu:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(device)
        else:
            device = torch.device("cpu")

        self.model.eval()
        losses = []
        with torch.no_grad():
            for X, y in loader:
                if use_gpu:
                    X = X.to(device)
                    y = y.to(device)
                output = self.model(X)
                loss = criterion(output, y)
                losses.append(loss.item())

        loss = np.mean(losses)
        return loss

    def train(self, criterion, train_loader, val_loader=None, epochs=10, use_gpu=False,
              steps_per_epoch=500, save_model_afer_every_epoch=5, lr_scheduler=None):

        """
        steps_per_epoch = should be around len(train_loader) / batch_size,
                        so that every example in the dataset gets convered in each epoch.
        """
        if use_gpu:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(device)
        else:
            device = torch.device("cpu")

        for epoch in range(self.epochs_completed, self.epochs_completed + epochs):
            train_losses = []
            # Training mode
            self.model.train()

            # Iterate over batches
            step = 0
            lr_step_done = False
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

                if isinstance(lr_scheduler, optim.lr_scheduler.CyclicLR):
                    lr_scheduler.step()
                    lr_step_done = True

                train_losses.append(loss.item())
                step = step + 1
                if step % steps_per_epoch == 0:
                    break

            train_loss = np.mean(train_losses)
            stats_info = 'Epoch: {}/{}\tTrain Loss: {:.6f}'.format(epoch + 1, epochs, train_loss)

            val_loss = np.inf
            if val_loader is not None:
                val_loss = self.evaluate(criterion, val_loader, use_gpu)
                stats_info = stats_info + "\tVal Loss: {:.6f}".format(val_loss)

                if isinstance(lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step(val_loss)
                    lr_step_done = True

                # Save best validation model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    print('Saving best validation model.')
                    self.save_model_optim_state('best_val_model.pt', epoch, train_loss=train_loss, val_loss=val_loss)

            if lr_scheduler is not None and not lr_step_done:
                lr_scheduler.step()
                lr_step_done = True

            # print training/validation statistics
            print(stats_info)
            if epoch % save_model_afer_every_epoch == 0:
                model_file_name = "model_epoch_{}.pt".format(epoch)
                self.save_model_optim_state(model_file_name, epoch, train_loss=train_loss, val_loss=val_loss)

        # Save latest model at the end
        self.save_model_optim_state("latest_model.pt", epoch, train_loss=train_loss, val_loss=val_loss)

    def predict_proba(self, loader, use_gpu=False):

        if use_gpu:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(device)
        else:
            device = torch.device("cpu")

        self.model.eval()
        preds = []
        y_true = []
        with torch.no_grad():
            for X, y in loader:
                if use_gpu:
                    X = X.to(device)
                pred = self.model(X).cpu()
                preds.append(pred)
                y_true.append(y)

        return torch.cat(preds), torch.cat(y_true)

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

        self.model.eval()
        with torch.no_grad():
            for iteration in range(iterations):
                print('Iteration:', iteration + 1)
                for X, y in tqdm_notebook(loader):
                    X = X.cuda()
                    feature_set = self.model(X).cpu().numpy()

                    if target_known:
                        y = y.numpy().reshape(-1,1)
                        feature_set = np.hstack([y, feature_set])

                    csv_writer.writerows(feature_set)
                    fp.flush()
        fp.close()

