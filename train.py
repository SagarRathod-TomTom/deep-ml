import os
import numpy as np
import torch
from tqdm.auto import tqdm
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

    def validate(self, criterion, loader, use_gpu=False):
        if loader is None:
            raise Exception('Loader cannot be None.')

        if use_gpu:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(device)
        else:
            device = torch.device("cpu")

        self.model.eval()
        running_loss = 0
        with torch.no_grad():
            bar = tqdm(total=len(loader), desc="{:12s}".format('Validation'))
            for batch_index, (X, y) in enumerate(loader):
                if use_gpu:
                    X = X.to(device)
                    y = y.to(device)
                output = self.model(X)
                loss = criterion(output, y)
                bar.update(1)
                running_loss = running_loss + ((loss.item() - running_loss) / (batch_index + 1))
                bar.set_postfix({'Val Loss': '{:.6f}'.format(running_loss)})

        return running_loss

    def fit(self, criterion, train_loader, val_loader=None, epochs=10, use_gpu=False,
              steps_per_epoch=None, save_model_afer_every_epoch=5, lr_scheduler=None):

        """
        steps_per_epoch = should be around len(train_loader) / batch_size,
                        so that every example in the dataset gets covered in each epoch.
        """
        if use_gpu:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(device)
        else:
            device = torch.device("cpu")

        if steps_per_epoch is None:
            steps_per_epoch = len(train_loader) + 1  # Number of batches + 1, since batch index starts with 0

        for epoch in range(self.epochs_completed, self.epochs_completed + epochs):

            print('Epoch {}/{}:'.format(epoch, self.epochs_completed + epochs))

            # Training mode
            self.model.train()

            # Iterate over batches
            step = 0
            lr_step_done = False
            running_train_loss = 0
            bar = tqdm(total=steps_per_epoch, desc="{:12s}".format('Training'))
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
                bar.update(1)
                running_train_loss = running_train_loss + ((loss.item() - running_train_loss) / step)
                bar.set_postfix({'Train loss': '{:.6f}'.format(running_train_loss)})
                if step % steps_per_epoch == 0:
                    bar.update(1)
                    break

            val_loss = np.inf
            if val_loader is not None:
                val_loss = self.validate(criterion, val_loader, use_gpu)

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

            if epoch % save_model_afer_every_epoch == 0:
                model_file_name = "model_epoch_{}.pt".format(epoch)
                self.save_model_optim_state(model_file_name, epoch, train_loss=running_train_loss, val_loss=val_loss)

        # Save latest model at the end
        self.save_model_optim_state("latest_model.pt", epoch, train_loss=running_train_loss, val_loss=val_loss)

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
            for X, y in tqdm(loader, total=len(loader), desc="{:12s}".format('Prediction')):
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
                for X, y in tqdm(loader, total=len(loader), desc='Feature Extraction'):
                    X = X.cuda()
                    feature_set = self.model(X).cpu().numpy()

                    if target_known:
                        y = y.numpy().reshape(-1,1)
                        feature_set = np.hstack([y, feature_set])

                    csv_writer.writerows(feature_set)
                    fp.flush()
        fp.close()
