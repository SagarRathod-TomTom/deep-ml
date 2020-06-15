import os
from abc import ABC, abstractmethod

import numpy as np
from tqdm.auto import tqdm
import torch
from .utils import binarize, plot_images


class Predictor(ABC):

    def __init__(self, model: torch.nn.Module, model_save_path=None, model_file_name='best_val_model.pt',
                 classes=None):
        super(Predictor, self).__init__()

        if model is None:
            raise ValueError('Model cannot be None.')

        self._model = model
        self.classes = classes

        if model_save_path and os.path.exists(os.path.join(model_save_path, model_file_name)):
            state_dict = torch.load(os.path.join(model_save_path, model_file_name))
            self._model.load_state_dict(state_dict['model'])

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
    def predict_one(self, input: torch.Tensor, use_gpu=False):
        pass

    @abstractmethod
    def show_predictions(self, loader, image_inverse_transform=None, samples=9, cols=3, figsize=(10, 10)):
        pass


class SemanticSegmentationPredictor(Predictor):

    def __init__(self, model: torch.nn.Module, model_save_path=None,
                 model_file_name=None, classes=None):
        super(SemanticSegmentationPredictor, self).__init__(model, model_save_path,
                                                            model_file_name, classes=classes)

    def predict_one(self, input: torch.Tensor, use_gpu=False):
        raise NotImplementedError

    def predict(self, loader, use_gpu=False):
        raise NotImplementedError()

    def show_predictions(self, loader, image_inverse_transform=None, samples=9, cols=3, figsize=(10, 10)):
        raise NotImplementedError()

    def transform_target(self, y, classes=None):
        raise NotImplementedError()

    def transform_output(self, prediction, classes=None):
        return binarize(prediction)


class ImageRegressionPredictor(Predictor):

    def __init__(self, model: torch.nn.Module, model_save_path=None,
                 model_file_name=None, classes=None):
        super(ImageRegressionPredictor, self).__init__(model, model_save_path,
                                                       model_file_name, classes=classes)

    def predict_one(self, input: torch.Tensor, use_gpu=False):
        raise NotImplementedError()

    def predict(self, loader, use_gpu=False):

        if len(loader) == 0:
            print('Loader is empty')
            return None

        device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
        self._model = self._model.to(device)
        predictions = []
        targets = []
        with torch.no_grad():
            for X, y in tqdm(loader, total=len(loader), desc="{:12s}".format('Prediction')):
                if use_gpu:
                    X = X.to(device)
                y_pred = self._model(X).cpu()
                predictions.append(y_pred)
                targets.append(y)

        predictions = torch.cat(predictions)
        targets = torch.cat(targets) if type(targets[0]) == torch.Tensor else np.hstack(targets).tolist()

        return predictions, targets

    def show_predictions(self, loader, image_inverse_transform=None, samples=9, cols=3, figsize=(10, 10)):

        self._model = self._model.to("cpu")
        self._model.eval()

        with torch.no_grad():
            indexes = np.random.randint(0, len(loader.dataset), samples)

            def transform(input_batch):
                x, y = input_batch
                prediction = self._model(x.unsqueeze(dim=0))
                return (self.transform_input(x, image_inverse_transform),
                        f'Ground Truth={self.transform_target(y)} '
                        f'\nPrediction={self.transform_output(prediction)}')

            image_title_generator = (transform(loader.dataset[index]) for index in indexes)
            plot_images(image_title_generator, samples=samples, cols=cols, figsize=figsize)

    def transform_target(self, y):
        return round(y.item(), 2)

    def transform_output(self, prediction):
        return round(prediction.item(), 2)


class ImageClassificationPredictor(ImageRegressionPredictor):

    def __init__(self, model: torch.nn.Module, model_save_path=None,
                 model_file_name=None, classes=None):
        super(ImageClassificationPredictor, self).__init__(model, model_save_path,
                                                           model_file_name, classes=classes)

    def predict_one(self, input: torch.Tensor, use_gpu=False):
        raise NotImplementedError()

    def predict(self, loader, use_gpu=False):
        predictions, targets = super(ImageClassificationPredictor, self).predict(loader,
                                                                                 use_gpu=use_gpu)

        return predictions, targets

    def transform_target(self, y):
        if self.classes:
            # if classes is not empty, replace target with actual class label
            y = self.classes[y]
        return y

    def transform_output(self, prediction):

        probability = prediction
        predicted_class = "-"

        if self.classes and prediction.shape[1] == len(self.classes):
            # multiclass
            probability, index = torch.max(prediction, dim=1)
            predicted_class = self.classes[index]
        elif self.classes and len(self.classes) == 2:
            # binary
            probability = prediction.item()
            predicted_class = probability >= 0.5 if self.classes[1] else self.classes[0]

        return predicted_class, probability

    def show_predictions(self, loader, image_inverse_transform=None, samples=9, cols=3, figsize=(10, 10)):

        self._model = self._model.to("cpu")
        self._model.eval()

        with torch.no_grad():
            indexes = np.random.randint(0, len(loader.dataset), samples)

            def transform(input_batch):
                x, y = input_batch
                prediction = self._model(x.unsqueeze(dim=0))
                predicted_class, probability = self.transform_output(prediction)

                return (self.transform_input(x, image_inverse_transform),
                        f'Ground Truth={self.transform_target(y)} '
                        f'\nPrediction={(predicted_class, round(probability.item(), 2))}')

            image_title_generator = (transform(loader.dataset[index]) for index in indexes)
            plot_images(image_title_generator, samples=samples, cols=cols, figsize=figsize)
