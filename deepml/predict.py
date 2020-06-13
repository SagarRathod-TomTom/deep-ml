import os
from abc import ABC, abstractmethod

import numpy as np
from tqdm.auto import tqdm
import torch
from .utils import binarize


class Predictor(ABC):

    def __init__(self, model: torch.nn.Module, model_save_path=None, model_file_name='best_val_model.pt'):
        super(Predictor, self).__init__()
        self.__model = model

        if model_save_path and os.path.exists(os.path.join(model_save_path, model_file_name)):
            state_dict = torch.load(os.path.join(model_save_path, model_file_name))
            self.__model.load_state_dict(state_dict['model'])

    def transform_input(self, x, image_inverse_transform=None):
        if image_inverse_transform is not None:
            x = image_inverse_transform(x.unsqueeze(dim=0)).squeeze()
        return x.numpy().transpose(1, 2, 0)  # CWH -> WHC

    @abstractmethod
    def transform_target(self, y):
        pass

    @abstractmethod
    def transform_output(self, prediction):
        pass

    @abstractmethod
    def predict(self, loader, use_gpu=False):
        pass

    @abstractmethod
    def predict_one(self,  input: torch.Tensor, use_gpu=False):
        pass

    @abstractmethod
    def show_predictions(self, loader, samples=9, image_inverse_transform=None, figsize=(10,10)):
        pass


class SemanticSegmentationPredictor(Predictor):

    def __init__(self, model: torch.nn.Module, model_save_path=None,
                 model_file_name=None, classes=None):
        super(SemanticSegmentationPredictor, self).__init__(model, model_save_path,
                                                            model_file_name)
        self.classes = classes

    def predict_one(self,  input: torch.Tensor, use_gpu=False):
        raise NotImplementedError

    def predict(self, loader, use_gpu=False):
        raise NotImplementedError()

    def show_predictions(self, loader, samples=9, image_inverse_transform=None, figsize=(10,10)):
        raise NotImplementedError()

    def transform_target(self, y):
        raise NotImplementedError()

    def transform_output(self, prediction):
        return binarize(prediction)


class ImageRegressionPredictor(Predictor):

    def __init__(self, model: torch.nn.Module, model_save_path=None,
                 model_file_name=None):
        super(ImageRegressionPredictor, self).__init__(model, model_save_path,
                                                       model_file_name)

    def predict_one(self, input: torch.Tensor, use_gpu=False):
        raise NotImplementedError()

    def predict(self, loader, use_gpu=False):

        if len(loader) == 0:
            print('Loader is empty')
            return None

        device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
        self.__model = self.__model.to(device)
        predictions = []
        targets = []
        with torch.no_grad():
            for X, y in tqdm(loader, total=len(loader), desc="{:12s}".format('Prediction')):
                if use_gpu:
                    X = X.to(device)
                y_pred = self.__model(X).cpu()
                predictions.append(y_pred)
                targets.append(y)

        predictions = torch.cat(predictions)
        targets = torch.cat(targets) if type(targets[0]) == torch.Tensor else np.hstack(targets).tolist()

        return predictions, targets

    def show_predictions(self, loader, samples=9, image_inverse_transform=None, figsize=(10,10)):
        pass

    def transform_target(self, y):
        return round(y.item(), 2)

    def transform_output(self, prediction):
        return round(prediction.item(), 2)


class ImageClassificationPredictor(ImageRegressionPredictor):

    def __init__(self, model: torch.nn.Module, model_save_path=None,
                 model_file_name=None, classes=None):
        super(ImageClassificationPredictor, self).__init__(model, model_save_path,
                                                           model_file_name)

        self.classes = classes

    def predict_one(self, input: torch.Tensor, use_gpu=False):
        raise NotImplementedError()

    def predict(self, loader, use_gpu=False):
        predictions, targets = super(ImageClassificationPredictor, self).predict(loader,
                                                                                 use_gpu=use_gpu)

        return predictions, targets

    def show_predictions(self, loader, samples=9, image_inverse_transform=None, figsize=(10,10)):
        pass

    def transform_target(self, y):
        if self.classes:
            # if classes is not empty, replace target with actual class label
            y = self.classes[y]
        return y

    def transform_output(self, prediction):

        probability = prediction
        predicted_class = None

        if self.classes and prediction.shape[1] == len(self.classes):
            # multiclass
            probability, index = torch.max(prediction, dim=1)
            predicted_class = self.classes[index]
        elif len(self.classes) == 2:
            # binary
            probability = prediction.item()
            predicted_class = probability >= 0.5 if self.classes[1] else self.classes[0]

        return predicted_class, probability
