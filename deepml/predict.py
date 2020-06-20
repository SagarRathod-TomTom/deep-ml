import os
from abc import ABC, abstractmethod

import numpy as np
from tqdm.auto import tqdm
import torch
import torchvision
from .utils import binarize, plot_images, create_text_image


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
                title_color = None
                return (self.transform_input(x, image_inverse_transform),
                        f'Ground Truth={self.transform_target(y)} '
                        f'\nPrediction={self.transform_output(prediction)}', title_color)

            image_title_generator = (transform(loader.dataset[index]) for index in indexes)
            plot_images(image_title_generator, samples=samples, cols=cols, figsize=figsize)

    def transform_target(self, y):
        return round(y.item(), 2)

    def transform_output(self, prediction):
        return round(prediction.item(), 2)

    def write_prediction_to_tensorboard(self, tag, image_batch, writer, image_inverse_transform,
                                        global_step, img_size=224):

        assert isinstance(img_size, int) or (isinstance(img_size, tuple) and len(img_size) == 2)

        if isinstance(img_size, int):
            img_size = (img_size, img_size)

        self._model.eval()
        with torch.no_grad():
            x, y = image_batch
            outputs = self._model(x).cpu()

            x, y = x.cpu(), y.cpu()
            if image_inverse_transform is not None:
                x = image_inverse_transform(x)

            input_img_size = tuple(x.shape[-2:])

            to_pillow_image = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                              torchvision.transforms.Resize(img_size)])
            to_tensor = torchvision.transforms.ToTensor()

            text = 'GT={ground_truth}\nPred={prediction}'
            output_images = []
            for index in range(x.shape[0]):
                ground_truth = self.transform_target(y[index])
                prediction = self.transform_output(outputs[index])
                content = text.format(ground_truth=ground_truth, prediction=prediction)
                content_image = create_text_image(content, img_size=img_size)

                if input_img_size != img_size:
                    output_images.append(to_tensor(to_pillow_image(x[index].squeeze(dim=0))))
                else:
                    output_images.append(x[index].squeeze(dim=0))
                output_images.append(to_tensor(content_image))

            writer.add_images(f'{tag}', torch.stack(output_images), global_step)


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
            probability = probability.item()
        elif self.classes and len(self.classes) == 2:
            # binary
            probability = prediction.item()
            predicted_class = probability >= 0.5 if self.classes[1] else self.classes[0]

        return predicted_class, round(probability, 2)

    def show_predictions(self, loader, image_inverse_transform=None, samples=9, cols=3, figsize=(10, 10)):

        self._model = self._model.to("cpu")
        self._model.eval()

        with torch.no_grad():
            indexes = np.random.randint(0, len(loader.dataset), samples)

            def transform(input_batch):
                x, y = input_batch
                prediction = self._model(x.unsqueeze(dim=0))
                predicted_class, probability = self.transform_output(prediction)
                target_class = self.transform_target(y)
                title_color = "green" if predicted_class == target_class else "red"
                return (self.transform_input(x, image_inverse_transform),
                        f'Ground Truth={target_class}'
                        f'\nPrediction={predicted_class}, {probability}', title_color)

            image_title_generator = (transform(loader.dataset[index]) for index in indexes)
            plot_images(image_title_generator, samples=samples, cols=cols, figsize=figsize)

    def write_prediction_to_tensorboard(self, tag, image_batch, writer, image_inverse_transform,
                                        global_step, img_size=224):

        assert isinstance(img_size, int) or (isinstance(img_size, tuple) and len(img_size) == 2)

        if isinstance(img_size, int):
            img_size = (img_size, img_size)

        self._model.eval()
        with torch.no_grad():
            x, y = image_batch
            outputs = self._model(x).cpu()

            x, y = x.cpu(), y.cpu()
            if image_inverse_transform is not None:
                x = image_inverse_transform(x)

            input_img_size = tuple(x.shape[-2:])
            to_pillow_image = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                              torchvision.transforms.Resize(img_size)])
            to_tensor = torchvision.transforms.ToTensor()

            text = 'GT={ground_truth}\nPred={predicted_class}, {probability}'
            output_images = []
            for index in range(x.shape[0]):
                ground_truth = self.transform_target(y[index])
                predicted_class, probability = self.transform_output(outputs[index].unsqueeze(dim=0))
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
