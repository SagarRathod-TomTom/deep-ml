import os
from abc import ABC, abstractmethod

import numpy as np
from tqdm.auto import tqdm
import torch
import torchvision
import torch.nn.functional as F

from .utils import binarize, plot_images, create_text_image, get_random_samples_batch_from_loader


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
        """
       Accepts input image batch in #BCHW form

       :param x: input image batch
       :param image_inverse_transform: an optional inverse transform to apply
       :return:
       """
        if image_inverse_transform is not None:
            x = image_inverse_transform(x)
        return x

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
    def predict_class(self, loader, use_gpu=False):
        pass

    @abstractmethod
    def show_predictions(self, loader, image_inverse_transform=None, samples=9, cols=3, figsize=(10, 10)):
        pass

    @abstractmethod
    def write_prediction_to_tensorboard(self, tag, image_batch, writer, image_inverse_transform,
                                        global_step, img_size=224):
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

    def predict_class(self, loader, use_gpu=False):
        raise NotImplementedError()

    def show_predictions(self, loader, image_inverse_transform=None, samples=9, cols=3, figsize=(10, 10)):
        raise NotImplementedError()

    def transform_target(self, y, classes=None):
        raise NotImplementedError()

    def transform_output(self, prediction, classes=None):
        return binarize(prediction)

    def write_prediction_to_tensorboard(self, tag, image_batch, writer, image_inverse_transform,
                                        global_step, img_size=224):
        raise NotImplementedError()


class ImageRegressionPredictor(Predictor):
    """
    The class useful for doing direct predictions in image classification problems.
    """

    def __init__(self, model: torch.nn.Module, model_save_path=None,
                 model_file_name=None, classes=None):
        super(ImageRegressionPredictor, self).__init__(model, model_save_path,
                                                       model_file_name, classes=classes)

    def predict(self, loader, use_gpu=False):
        """
        Accepts torch data loader and performs prediction
        :param loader:
        :param use_gpu: boolean indicating whether to use GPU
        :return: triplet of torch.Tensor of (targets, predicted class index, probability)
        """

        assert loader is not None and len(loader) > 0

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
        targets = torch.cat(targets) if isinstance(targets[0], torch.Tensor) else np.hstack(targets).tolist()

        return predictions, targets

    def predict_class(self, loader, use_gpu=False):
        raise NotImplementedError()

    def show_predictions(self, loader, image_inverse_transform=None, samples=9, cols=3, figsize=(10, 10)):
        """
        Shows predictions of random samples from loader. Draws matplotlib figure.
        :param loader: the torch data loader
        :param image_inverse_transform: the inverse transform for image
        :param samples: the number of samples. Default is 9
        :param cols: the number of columns to use while displaying images. Default is 3
        :param figsize: the matplotlib figure size to use. Default is 10, 10
        :return:
        """

        self._model = self._model.to("cpu")
        self._model.eval()

        with torch.no_grad():
            x, y = get_random_samples_batch_from_loader(loader, samples)
            predictions = self._model(x)

            x = self.transform_input(x, image_inverse_transform)
            # #BCHW --> #BHWC
            x = x.permute([0, 2, 3, 1])
            title_color = None

            def create_title(y, prediction):
                return f'Ground Truth={self.transform_target(y)}' \
                       f'\nPrediction={self.transform_output(prediction)}'

            image_title_generator = ((x[index], create_title(y[index], predictions[index]),
                                      title_color)
                                     for index in range(x.shape[0]))

            plot_images(image_title_generator, samples=samples, cols=cols, figsize=figsize)

    def transform_target(self, y):
        """
        Accepts python float
        :param y:
        :return:
        """
        return round(y.item(), 2)

    def transform_output(self, prediction):
        """
        Accepts python float
        :param prediction:
        :return:
        """
        return round(prediction.item(), 2)

    def write_prediction_to_tensorboard(self, tag, image_batch, writer, image_inverse_transform,
                                        global_step, img_size=224):
        """
        Writes prediction to TensorBoard

        :param tag: unique tag
        :param image_batch: input image batch with corresponding target (X,y)
        :param writer: tensorboard writer object
        :param image_inverse_transform: reverse image transform
        :param global_step: the epoch value
        :param img_size: image size to use while writing image to tensorboard. Default is 224.
        :return: None
        """

        assert isinstance(img_size, int) or (isinstance(img_size, tuple) and len(img_size) == 2)

        if isinstance(img_size, int):
            img_size = (img_size, img_size)

        self._model.eval()
        with torch.no_grad():
            x, y = image_batch
            predictions = self._model(x).cpu()

            x, y = x.cpu(), y.cpu()
            x = self.transform_input(x, image_inverse_transform)
            input_img_size = tuple(x.shape[-2:])

            to_pillow_image = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                              torchvision.transforms.Resize(img_size)])
            to_tensor = torchvision.transforms.ToTensor()

            text = 'GT={ground_truth}\nPred={prediction}'
            output_images = []
            for index in range(x.shape[0]):
                ground_truth = self.transform_target(y[index])
                prediction = self.transform_output(predictions[index])
                content = text.format(ground_truth=ground_truth, prediction=prediction)
                content_image = create_text_image(content, img_size=img_size)

                if input_img_size != img_size:
                    output_images.append(to_tensor(to_pillow_image(x[index].squeeze(dim=0))))
                else:
                    output_images.append(x[index].squeeze(dim=0))
                output_images.append(to_tensor(content_image))

            writer.add_images(f'{tag}', torch.stack(output_images), global_step)


class ImageClassificationPredictor(ImageRegressionPredictor):
    """
    The class useful for doing direct predictions in image classification problems.
    """

    def __init__(self, model: torch.nn.Module, model_save_path=None,
                 model_file_name=None, classes=None):
        super(ImageClassificationPredictor, self).__init__(model, model_save_path,
                                                           model_file_name, classes=classes)

    def predict(self, loader, use_gpu=False):
        """
        Accepts torch data loader and performs prediction
        :param loader:
        :param use_gpu: boolean indicating whether to use GPU
        :return: triplet of torch.Tensor of (targets, predicted class index, probability)
        """

        assert loader is not None and len(loader) > 0
        device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
        self._model = self._model.to(device)
        predictions = []
        targets = []

        with torch.no_grad():
            for X, y in tqdm(loader, total=len(loader), desc="{:12s}".format('Prediction')):
                X = X.to(device)
                y_pred = self._model(X).cpu()
                predictions.append(y_pred)
                targets.append(y)

        predictions = torch.cat(predictions)

        targets = torch.cat(targets) if isinstance(targets[0], torch.Tensor) else np.hstack(targets).tolist()

        return targets, predictions

    def predict_class(self, loader, use_gpu=False):
        targets, predictions = self.predict(loader, use_gpu)
        indices, probability = self.transform_output(predictions)
        return targets, indices, probability

    def transform_target(self, y):
        if self.classes:
            # if classes is not empty, replace target with actual class label
            y = self.classes[y]
        return y

    def transform_output(self, predictions):
        """
        Accepts batch of predictions and applies either sigmoid or softmax based on
        the type of classification
        :param predictions:
        :return:
        """

        if predictions.shape[-1] > 1:
            # multiclass
            probability, indices = torch.max(F.softmax(predictions, dim=1), dim=1)
        else:
            # binary
            probability = torch.sigmoid(predictions)
            indices = torch.zeros_like(probability)
            indices[probability > 0.5] = 1

        return indices, probability

    def show_predictions(self, loader, image_inverse_transform=None, samples=9, cols=3, figsize=(10, 10)):
        """
        Shows predictions of random samples from loader. Draws matplotlib figure.
        :param loader: the torch data loader
        :param image_inverse_transform: the inverse transform for image
        :param samples: the number of samples. Default is 9
        :param cols: the number of columns to use while displaying images. Default is 3
        :param figsize: the matplotlib figure size to use. Default is 10, 10
        :return:
        """

        self._model = self._model.to("cpu")
        self._model.eval()

        with torch.no_grad():
            x, targets = get_random_samples_batch_from_loader(loader, samples)
            predictions = self._model(x.to("cpu"))

            x = self.transform_input(x, image_inverse_transform)
            # #BCHW --> #BHWC
            x = x.permute([0, 2, 3, 1])

            class_indices, probabilities = self.transform_output(predictions)

            def create_title(y, class_index, probability):
                target_class = self.transform_target(y)
                predicted_class = self.transform_target(class_index)
                title_color = "green" if predicted_class == target_class else "red"
                return (f'Ground Truth={target_class}'
                        f'\nPrediction={predicted_class}, '
                        f'{round(probability.item(), 2)}', title_color)

            image_title_generator = ((x[index], *create_title(targets[index], class_indices[index],
                                                              probabilities[index]))
                                     for index in range(x.shape[0]))

            plot_images(image_title_generator, samples=samples, cols=cols, figsize=figsize)

    def write_prediction_to_tensorboard(self, tag, image_batch, writer, image_inverse_transform,
                                        global_step, img_size=224):
        """
        Writes prediction to TensorBoard

        :param tag: unique tag
        :param image_batch: input image batch with corresponding target (X,y)
        :param writer: tensorboard writer object
        :param image_inverse_transform: reverse image transform
        :param global_step: the epoch value
        :param img_size: image size to use while writing image to tensorboard. Default is 224.
        :return: None
        """

        assert isinstance(img_size, int) or (isinstance(img_size, tuple) and len(img_size) == 2)

        if isinstance(img_size, int):
            img_size = (img_size, img_size)

        self._model.eval()
        with torch.no_grad():
            x, y = image_batch
            predictions = self._model(x).cpu()

            x = self.transform_input(x).cpu()
            class_indices, probabilities = self.transform_output(predictions)

            input_img_size = tuple(x.shape[-2:])
            to_pillow_image = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                              torchvision.transforms.Resize(img_size)])
            to_tensor = torchvision.transforms.ToTensor()

            text = '{ground_truth}\n{predicted_class}, {probability}'
            output_images = []
            for index in range(x.shape[0]):
                ground_truth = self.transform_target(y[index])
                predicted_class = self.transform_target(class_indices[index])
                probability = round(probabilities[index].item(), 2)

                content = text.format(ground_truth=ground_truth, predicted_class=predicted_class,
                                      probability=probability)
                text_color = "green" if ground_truth == predicted_class else "red"
                content_image = create_text_image(content, img_size=img_size, text_color=text_color)

                if input_img_size != img_size:
                    output_images.append(to_tensor(to_pillow_image(x[index].squeeze(dim=0))))
                else:
                    output_images.append(x[index].squeeze(dim=0))

                output_images.append(to_tensor(content_image))

            writer.add_images(f'{tag}', torch.stack(output_images), global_step)
