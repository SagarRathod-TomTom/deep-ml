class Trainer:
    def __init__(self, model, optimizer, model_save_path, load_saved_model=False, 
                 model_file_name='latest_model.pt'):
        self.model = model
        self.optimizer = optimizer
        self.model_save_path = model_save_path
        self.epochs_completed = 0
        self.best_val_loss = np.inf
        
        if load_saved_model:
            print('Loading Saved Model Weights.')
            state_dict = torch.load(os.path.join(self.model_save_path, model_file_name))
            self.model.load_state_dict(state_dict['model'])
            self.optimizer.load_state_dict(state_dict['optimizer'])
            self.epochs_completed = state_dict['epoch']
            self.best_val_loss = state_dict['metrics']['val_loss']
    
    def save_model_optim_state(self, model_file_name, epoch, train_loss, val_loss):
        torch.save({'model' : self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                    'metrics' : {'train_loss' : train_loss, 'val_loss' : val_loss},
                    'epoch' : epoch
                   }, 
                   os.path.join(self.model_save_path, model_file_name)
                  )
        
    def evaluate(self, criterion, loader, use_gpu=False):
        if loader is None:
            raise Exception('Loader can not be None.')

        if use_gpu:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(device)
        
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch_index, (X, y) in enumerate(loader):
                if use_gpu:
                    X = X.to(device)
                    y = y.to(device)

                    ## update the loss
                    output = self.model(X)
                    loss = criterion(output, y)
                    losses.append(loss.item())
        loss = np.mean(losses)
        return loss

    def train(self, criterion, train_loader, val_loader=None, epochs=10, use_gpu=False,
             steps_per_epoch=500, save_model_afer_every_epoch=5):
        
        """
        steps_per_epoch = should be around len(train_loader) / batch_size,
                        so that every example in the dataset gets convered in each epoch.
        """    
        if use_gpu:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(device)
    

        for epoch in range(self.epochs_completed, self.epochs_completed + epochs):
            
            train_losses = []
            # Training mode
            self.model.train()
            
            #Iterate over batches
            step = 0
            for batch_index, (X, y) in enumerate(train_loader):
                
                if use_gpu:
                    X = X.to(device)
                    y = y.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                outputs = self.model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
                step = step + 1
                if step % steps_per_epoch == 0:
                    break
            
            train_loss = np.mean(train_losses)
            stats_info = 'Epoch: {}/{}\tTrain Loss: {:.6f}'.format(epoch + 1, epochs,
                                                                  train_loss)

            if val_loader is not None:
                val_loss = self.evaluate(criterion, val_loader, use_gpu)
                stats_info = stats_info + "\tVal Loss: {:.6f}".format(val_loss)

                # Save best validation model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    print('Saving best validation model.')
                    self.save_model_optim_state('best_val_model.pt', epoch, train_loss=train_loss, 
                                            val_loss=val_loss)
            
            # print training/validation statistics 
            print(stats_info)
            if epoch % save_model_afer_every_epoch == 0:
                model_file_name = "model_epoch_{}.pt".format(epoch)
                self.save_model_optim_state(model_file_name, epoch, train_loss=train_loss, 
                                           val_loss=val_loss)
        
        # Save latest model at the end
        self.save_model_optim_state("latest_model.pt", epoch, train_loss=train_loss, 
                                    val_loss=val_loss)
        
    def predict_proba(self, loader, use_gpu=False):

        if use_gpu:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(device)
        
        self.model.eval()
        preds = []
        with torch.no_grad():
            for X, index in loader:

                if use_gpu:
                    X = X.to(device)

                pred = self.model(X).cpu()
                preds.append(pred)
            
        return torch.cat(preds, axis=0)
