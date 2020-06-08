import os
from abc import ABC, abstractmethod

import numpy as np
from tqdm.auto import tqdm
import torch


class Predictor(ABC):

    def __init__(self, model: torch.nn.Module, model_save_path: str, model_file_name='best_val_model.pt'):
        super(Predictor, self).__init__()
        state_dict = torch.load(os.path.join(model_save_path, model_file_name))
        self.model = model
        self.model.load_state_dict(state_dict['model'])

    @abstractmethod
    def predict(self, loader, use_gpu=False):
        pass

    @abstractmethod
    def predict_one(self, input: torch.Tensor, use_gpu=False):
        pass


class SemanticSegmentationPredictor(Predictor):

    def __init__(self, model: torch.nn.Module, model_save_path: str,
                 model_file_name: str):
        super(SemanticSegmentationPredictor, self).__init__(model, model_save_path,
                                                            model_file_name)

    def predict_one(self, input: torch.Tensor, use_gpu=False):
        raise NotImplementedError

    def predict(self, loader, use_gpu=False):
        raise NotImplementedError()


class ImageRegressionPredictor(Predictor):

    def __init__(self, model: torch.nn.Module, model_save_path: str,
                 model_file_name: str):
        super(ImageRegressionPredictor, self).__init__(model, model_save_path,
                                                       model_file_name)

    def predict_one(self, input: torch.Tensor, use_gpu=False):
        raise NotImplementedError()

    def predict(self, loader, use_gpu=False):

        if len(loader) == 0:
            print('Loader is empty')
            return None

        device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        predictions = []
        targets = []
        with torch.no_grad():
            for X, y in tqdm(loader, total=len(loader), desc="{:12s}".format('Prediction')):
                if use_gpu:
                    X = X.to(device)
                y_pred = self.model(X).cpu()
                predictions.append(y_pred)
                targets.append(y)

        predictions = torch.cat(predictions)
        targets = torch.cat(targets) if type(targets[0]) == torch.Tensor else np.hstack(targets).tolist()

        return predictions, targets


class ImageClassificationPredictor(Predictor):

    def __init__(self, model: torch.nn.Module, model_save_path: str,
                 model_file_name: str):
        super(ImageClassificationPredictor, self).__init__(model, model_save_path,
                                                           model_file_name)

    def predict_one(self, input: torch.Tensor, use_gpu=False):
        raise NotImplementedError()

    def predict(self, loader, use_gpu=False):
        raise NotImplementedError()
