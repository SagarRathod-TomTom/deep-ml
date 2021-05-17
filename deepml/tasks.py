import os
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from tqdm.auto import tqdm

from deepml.utils import create_text_image, get_random_samples_batch_from_loader
from deepml.visualize import plot_images_with_title, plot_images


class Predictor(ABC):

    def __init__(self, model: torch.nn.Module, model_dir, load_saved_model=False,
                 model_file_name='latest_model.pt', use_gpu=True):

        super(Predictor, self).__init__()

        assert isinstance(model, torch.nn.Module)
        assert model_dir is not None
        assert isinstance(model_file_name, str)

        self._model = model
        self._device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

        self._model_dir = model_dir
        self._model_file_name = model_file_name

        os.makedirs(self.model_dir, exist_ok=True)
        weights_file_path = os.path.join(self._model_dir, self._model_file_name)
        if load_saved_model and os.path.exists(weights_file_path):
            self.__load_model_weights(weights_file_path)

    def __load_model_weights(self, weights_file_path):
        if weights_file_path and os.path.exists(weights_file_path):
            print(f'Loading Saved Model Weights: {weights_file_path}')
            state_dict = (torch.load(weights_file_path) if torch.cuda.is_available() else
                          torch.load(weights_file_path, map_location=torch.device('cpu')))
            self._model.load_state_dict(state_dict['model'])

    @property
    def model(self):
        return self._model

    @property
    def device(self):
        return self._device

    @property
    def model_dir(self):
        return self._model_dir

    @property
    def model_file_name(self):
        return self._model_file_name

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
    def predict_batch(self, x):
        pass

    @abstractmethod
    def predict(self, loader):
        pass

    @abstractmethod
    def predict_class(self, loader):
        pass

    @abstractmethod
    def show_predictions(self, loader, image_inverse_transform=None, samples=9, cols=3, figsize=(10, 10)):
        pass

    @abstractmethod
    def write_prediction_to_tensorboard(self, tag, loader, writer, image_inverse_transform,
                                        global_step, img_size=224):
        pass


class NeuralNetPredictor(Predictor):
    """
        Use this simple predictor class for any deep learning task.
        It avoids writing to tensorboard and does not apply any transformation
        on input and output.
    """

    def __init__(self, model: torch.nn.Module, model_dir, load_saved_model=False,
                 model_file_name='latest_model.pt', use_gpu=True):
        super(NeuralNetPredictor, self).__init__(model, model_dir, load_saved_model,
                                                 model_file_name, use_gpu)

    def predict_batch(self, x):
        x = x.to(self._device)
        return self._model(x)

    def predict(self, loader):
        """
        Accepts torch data loader and performs prediction
        :param loader:
        :return: tuple of torch.Tensor of (prediction, targets)
        """

        assert loader is not None and len(loader) > 0
        self._model = self._model.to(self._device)
        predictions = []
        targets = []
        with torch.no_grad():
            for x, y in tqdm(loader, total=len(loader), desc="{:12s}".format('Prediction')):
                y_pred = self.predict_batch(x).cpu()
                predictions.append(y_pred)
                targets.append(y)

        predictions = torch.cat(predictions)
        targets = torch.cat(targets) if isinstance(targets[0], torch.Tensor) else np.hstack(targets).tolist()

        return predictions, targets

    def predict_class(self, loader):
        raise NotImplementedError()

    def show_predictions(self, loader, image_inverse_transform=None, samples=9, cols=3, figsize=(10, 10)):
        raise NotImplementedError()

    def transform_target(self, y):
        raise NotImplementedError()

    def transform_output(self, prediction):
        raise NotImplementedError()

    def write_prediction_to_tensorboard(self, tag, loader, writer, image_inverse_transform,
                                        global_step, img_size=224):
        pass


class Segmentation(NeuralNetPredictor):
    """
    This class is useful for binary and Multiclass Segmentation.
    """

    def __init__(self, model: torch.nn.Module, model_dir, load_saved_model=False,
                 model_file_name='latest_model.pt', use_gpu=True, classes=None, threshold=0.5):
        super(Segmentation, self).__init__(model, model_dir, load_saved_model,
                                           model_file_name, use_gpu)

        assert isinstance(classes, int), "should be the number of classes"
        assert classes > 1, "for binary segmentation task, it should be 2 classes"

        self.classes = classes
        self.threshold = threshold

        if self.classes == 2:
            self.color_map = {0: 0, 1: 255}
        else:
            self.color_map = {0: [[0, 0, 0]]}
            additional_colors = np.random.randint(0, 256, size=(self.classes - 1, 3))
            for index, color in enumerate(additional_colors.tolist()):
                self.color_map[index + 1] = color

    def predict_batch(self, x):
        x = x.to(self._device)
        pred = self._model(x)

        if isinstance(pred, dict) and 'out' in pred:
            return pred['out']  # torchvision model's returns prediction in OrderedDict
        else:
            return pred

    def predict(self, loader, **kwargs):
        """
        Accepts torch data loader and performs prediction
        :param loader: torch.data loaders
        :param save_dir : the output path to save predicted segmentation mask
        :return: tuple of torch.Tensor of (prediction, targets)
        """
        assert loader is not None and len(loader) > 0
        save_dir = kwargs.get('save_dir', None)

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        predictions = []
        targets = []

        self._model = self._model.to(self._device)
        with torch.no_grad():
            for x, y in tqdm(loader, total=len(loader), desc="{:12s}".format('Prediction')):
                y_pred = self.predict_batch(x).cpu()

                if save_dir is not None:
                    output_mask = self.decode_segmentation_mask(self.transform_output(y_pred))
                    Segmentation.save_image_batch(output_mask, save_dir, y.tolist())
                else:
                    predictions.append(y_pred)
                    targets.append(y)

        if len(predictions) > 0 and len(targets) > 0:
            predictions = torch.cat(predictions)
            targets = torch.cat(targets) if isinstance(targets[0], torch.Tensor) else np.hstack(targets).tolist()

        return predictions, targets

    @staticmethod
    def save_image_batch(output_mask, outdir, filenames):
        assert output_mask.ndim == 4, "should be in the form of BCHW"

        for i in range(output_mask.shape[0]):
            image = torchvision.transforms.functional.to_pil_image(output_mask[i])
            image.save(os.path.join(outdir, filenames[i]))

    def predict_class(self, loader):
        raise NotImplementedError()

    def show_predictions(self, loader, image_inverse_transform=None, samples=4, cols=3, figsize=(16, 16)):
        self._model = self._model.to(self._device)
        self._model.eval()

        with torch.no_grad():
            x, targets = get_random_samples_batch_from_loader(loader, samples)
            predictions = self.predict_batch(x).cpu()

            x = self.transform_input(x, image_inverse_transform)
            target_mask = self.decode_segmentation_mask(targets)
            class_indices = self.transform_output(predictions)
            output_mask = self.decode_segmentation_mask(class_indices)

            # BCHW --> #BHWC
            x = x.permute([0, 2, 3, 1])
            target_mask = target_mask.permute([0, 2, 3, 1])
            output_mask = output_mask.permute([0, 2, 3, 1])

            images = []
            for i in range(x.shape[0]):
                images.extend([x[i], target_mask[i], output_mask[i]])

            image_titles = ["Input", "Target", "Prediction"] * x.shape[0]
            plot_images(images, image_titles, cols=cols, figsize=figsize, fontsize=12)
        return x, class_indices

    def transform_target(self, y):
        return self.decode_segmentation_mask(y)

    def transform_output(self, predictions):

        assert predictions.ndim == 4  # B,C,H,W

        if predictions.shape[1] == 1:
            # Binary
            probability = torch.sigmoid(predictions)
            class_indices = torch.zeros_like(probability)
            class_indices[probability >= self.threshold] = 1
        else:
            # Multiclass
            probability = torch.softmax(predictions, dim=1)
            class_indices = torch.argmax(probability, dim=1)

        return class_indices

    def decode_segmentation_mask(self, class_indices):
        assert class_indices.ndim == 3  # B,H,W

        decoded_images = []
        out_channel = 3 if self.classes > 2 else 1

        # For each image in the batch
        for i in range(class_indices.shape[0]):
            output_mask = np.zeros((*class_indices[i].shape, out_channel), dtype=np.uint8)  # H,W, C
            for label in class_indices[i].unique():
                idx = class_indices[i] == label
                output_mask[idx] = self.color_map[label.item()]
            decoded_images.append(torch.from_numpy(output_mask.transpose(2, 0, 1)))

        return torch.stack(decoded_images)

    def write_prediction_to_tensorboard(self, tag, loader, writer, image_inverse_transform,
                                        global_step, img_size=224):

        self._model = self._model.to(self._device)
        self._model.eval()

        with torch.no_grad():
            x, targets = get_random_samples_batch_from_loader(loader)
            predictions = self.predict_batch(x).cpu()

            x = self.transform_input(x, image_inverse_transform)
            target_mask = self.decode_segmentation_mask(targets)
            class_indices = self.transform_output(predictions)
            output_mask = self.decode_segmentation_mask(class_indices)

            target_mask = target_mask.to(x.dtype)
            output_mask = output_mask.to(x.dtype)

            images = []
            for i in range(x.shape[0]):
                images.extend([x[i], target_mask[i], output_mask[i]])

            writer.add_images(tag, torch.stack(images), global_step)


class ImageRegression(NeuralNetPredictor):
    """
    The class useful to perform image regression.
    """

    def __init__(self, model: torch.nn.Module, model_dir, load_saved_model=False,
                 model_file_name='latest_model.pt', use_gpu=True):
        super(ImageRegression, self).__init__(model, model_dir, load_saved_model,
                                              model_file_name, use_gpu)

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

        self._model = self._model.to(self._device)
        self._model.eval()

        with torch.no_grad():
            x, y = get_random_samples_batch_from_loader(loader, samples)
            predictions = self.predict_batch(x)

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

            plot_images_with_title(image_title_generator, samples=samples, cols=cols, figsize=figsize)

    def transform_target(self, y):
        """
        Accepts torch Tensor
        :param y:
        :return:
        """
        return round(y.item(), 2)

    def transform_output(self, prediction):
        """
        Accepts torch Tensor
        :param prediction:
        :return:
        """
        return round(prediction.item(), 2)

    def write_prediction_to_tensorboard(self, tag, loader, writer, image_inverse_transform,
                                        global_step, img_size=224):
        """
        Writes prediction to TensorBoard

        :param tag: unique tag
        :param loader: the torch data loader
        :param writer: tensorboard writer object
        :param image_inverse_transform: reverse image transform
        :param global_step: the epoch value
        :param img_size: image size to use while writing image to tensorboard. Default is 224.
        :return: None
        """

        assert isinstance(img_size, int) or (isinstance(img_size, tuple) and len(img_size) == 2)

        if isinstance(img_size, int):
            img_size = (img_size, img_size)

        self._model = self._model.to(self._device)
        self._model.eval()
        with torch.no_grad():
            x, targets = get_random_samples_batch_from_loader(loader)
            predictions = self.predict_batch(x).cpu()

            x, y = x.cpu(), targets.cpu()
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

    def predict_class(self, loader):
        raise NotImplementedError()


class ImageClassification(NeuralNetPredictor):
    """
    The class useful for image classification task.
    """

    def __init__(self, model: torch.nn.Module, model_dir, load_saved_model=False,
                 model_file_name='latest_model.pt', use_gpu=True, classes=None):
        super(ImageClassification, self).__init__(model, model_dir, load_saved_model,
                                                  model_file_name, use_gpu)
        self._classes = classes

    def predict_class(self, loader):
        predictions, targets = self.predict(loader)
        predicted_class, probability = self.transform_output(predictions)
        return predicted_class, probability, targets

    def transform_target(self, y):
        if self._classes:
            # if classes is not empty, replace target with actual class label
            y = self._classes[y]
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

        self._model = self._model.to(self._device)
        self._model.eval()

        with torch.no_grad():
            x, targets = get_random_samples_batch_from_loader(loader, samples)
            predictions = self.predict_batch(x).cpu()

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

            plot_images_with_title(image_title_generator, samples=samples, cols=cols, figsize=figsize)

    def write_prediction_to_tensorboard(self, tag, loader, writer, image_inverse_transform,
                                        global_step, img_size=224):
        """
        Writes prediction to TensorBoard

        :param tag: unique tag
        :param loader: the torch data loader
        :param writer: tensorboard writer object
        :param image_inverse_transform: reverse image transform
        :param global_step: the epoch value
        :param img_size: image size to use while writing image to tensorboard. Default is 224.
        :return: None
        """

        assert isinstance(img_size, int) or (isinstance(img_size, tuple) and len(img_size) == 2)

        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self._model = self._model.to(self._device)
        self._model.eval()
        with torch.no_grad():
            x, targets = get_random_samples_batch_from_loader(loader)
            predictions = self.predict_batch(x).cpu()

            x = self.transform_input(x).cpu()
            class_indices, probabilities = self.transform_output(predictions)

            input_img_size = tuple(x.shape[-2:])
            to_pillow_image = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                              torchvision.transforms.Resize(img_size)])
            to_tensor = torchvision.transforms.ToTensor()

            text = '{ground_truth}\n{predicted_class}, {probability}'
            output_images = []
            for index in range(x.shape[0]):
                ground_truth = self.transform_target(targets[index])
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


class MultiLabelImageClassification(ImageClassification):
    """
    The class useful for multi label image classification task.
    """

    def __init__(self, model: torch.nn.Module, model_dir, load_saved_model=False,
                 model_file_name='latest_model.pt', use_gpu=True, classes=None):
        super(ImageClassification, self).__init__(model, model_dir, load_saved_model,
                                                  model_file_name, use_gpu)
        self._classes = classes

    def predict_class(self, loader):
        predictions, targets = self.predict(loader)
        predicted_class, probability = self.transform_output(predictions)
        return predicted_class, probability, targets

    def transform_target(self, y):
        if self._classes:
            # if classes is not empty, replace target with actual class label
            y = ", ".join([self._classes[index] for index, value in enumerate(y) if value])
        return y

    def transform_output(self, predictions):
        """
        Accepts batch of predictions and applies either sigmoid or softmax based on
        the type of classification
        :param predictions:
        :return:
        """
        probability = torch.sigmoid(predictions)
        indices = torch.zeros_like(probability)
        indices[probability > 0.5] = 1

        return indices, probability
