import os
from abc import ABC, abstractmethod
from typing import Union, Tuple, Any, List, Sequence, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision
from PIL import Image
from tqdm.auto import tqdm

from deepml.tracking import MLExperimentLogger
from deepml.utils import create_text_image, get_random_samples_batch_from_loader
from deepml.visualize import plot_images_with_title, plot_images


class Task(ABC):

    def __init__(self, model: torch.nn.Module, model_dir: str, load_saved_model: bool = False,
                 model_file_name: str = 'latest_model.pt', use_gpu: bool = True):

        super(Task, self).__init__()

        assert isinstance(model, torch.nn.Module)
        assert model_dir is not None
        assert isinstance(model_file_name, str)

        self._model = model
        self._device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

        self._model_dir = model_dir
        self._model_file_name = model_file_name

        os.makedirs(self.model_dir, exist_ok=True)
        weights_file_path = os.path.join(self._model_dir, self._model_file_name)
        if load_saved_model:
            self.__load_model_weights(weights_file_path)

    def __load_model_weights(self, weights_file_path: str):
        if weights_file_path and os.path.exists(weights_file_path):
            print(f'Loading Saved Model Weights: {weights_file_path}')
            state_dict = (torch.load(weights_file_path) if torch.cuda.is_available() else
                          torch.load(weights_file_path, map_location=torch.device('cpu')))
            self._model.load_state_dict(state_dict['model'])
            print("Model Weights Successfully Loaded!")

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

    def move_input_to_device(self, x: Union[torch.Tensor, list, tuple], non_blocking=False):
        if isinstance(x, torch.Tensor):
            x = x.to(self._device, non_blocking=non_blocking)
        elif isinstance(x, list):  # list of torch tensors
            x = [i.to(self._device, non_blocking=non_blocking) for i in x]
        elif isinstance(x, tuple):  # tuple of torch tensors
            x = tuple([i.to(self._device, non_blocking=non_blocking) for i in x])
        elif isinstance(x, dict):  # dict values as torch tensors
            x = {key: value.to(self._device, non_blocking=non_blocking) for key, value in x.items()}

        return x

    def transform_input(self, x: torch.Tensor, image_inverse_transform: Callable = None):
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
    def predict_batch(self, x, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, loader):
        pass

    @abstractmethod
    def predict_class(self, loader):
        pass

    @abstractmethod
    def show_predictions(self, loader, image_inverse_transform=None, samples=9, cols=3, figsize=(10, 10),
                         target_known=True):
        pass

    @abstractmethod
    def write_prediction_to_logger(self, tag, loader, logger, image_inverse_transform,
                                   global_step, img_size=224):
        pass


class NeuralNetTask(Task):
    """
        Use this simple predictor class for any deep learning task.
        It avoids writing to tensorboard and does not apply any transformation
        on input and output.
    """

    def __init__(self, model: torch.nn.Module, model_dir: str, load_saved_model: bool = False,
                 model_file_name: str = 'latest_model.pt', use_gpu: bool = True):
        super(NeuralNetTask, self).__init__(model, model_dir, load_saved_model,
                                            model_file_name, use_gpu)

    def predict_batch(self, x: torch.Tensor, *args, **kwargs):
        x = self.move_input_to_device(x, *args, **kwargs)
        return self._model(x)

    def predict(self, loader: torch.utils.data.DataLoader):
        """
        Accepts torch data loader and performs prediction
        :param loader:
        :return: tuple of torch.Tensor of (prediction, targets)
        """

        assert loader is not None and len(loader) > 0
        self._model.eval()
        self._model = self._model.to(self._device)
        predictions = []
        targets = []
        with torch.no_grad():
            for x, y in tqdm(loader, total=len(loader), desc="{:12s}".format('Prediction')):
                y_pred = self.predict_batch(x)
                predictions.append(y_pred)
                targets.append(y)

        predictions = torch.cat(predictions)
        targets = torch.cat(targets) if isinstance(targets[0], torch.Tensor) else np.hstack(targets).tolist()

        return predictions, targets

    def predict_class(self, loader: torch.utils.data.DataLoader):
        raise NotImplementedError()

    def show_predictions(self, loader: torch.utils.data.DataLoader,
                         image_inverse_transform: Callable = None, samples: int = 9, cols: int = 3,
                         figsize: Tuple[int, int] = (10, 10), target_known: bool = True):
        raise NotImplementedError()

    def transform_target(self, y: Any):
        raise NotImplementedError()

    def transform_output(self, prediction):
        raise NotImplementedError()

    def write_prediction_to_logger(self, tag, loader, logger, image_inverse_transform,
                                   global_step, img_size=224):
        pass


class Segmentation(NeuralNetTask):
    """
    This class is useful for binary and Multiclass pixel segmentation.
    """

    def __init__(self, model: torch.nn.Module, model_dir: str, load_saved_model: bool = False,
                 model_file_name: str = 'latest_model.pt', use_gpu: bool = True, num_classes: int = 1,
                 threshold: float = 0.5, color_map: dict = None):
        """

        :param model: The model architecture defined using torch.nn.Module
        :param model_dir:
        :param load_saved_model:
        :param model_file_name:
        :param use_gpu:
        :param num_classes:   The number of classes. Default is 1 for binary segmentation.
                          The class index 0 being background class and 1 being object of interest.
        :param threshold: The threshold for binary segmentation.
        :param color_map: The color map dictionary with class-index as a key and value as color.
                          If color_map is None,
                          For binary segmentation, by default it is {0: 0, 1: 255}
                          For multiclass segmentation, dict value should be RGB color triplet,
                          if not specified, random RGB color triplets are picked up.
                          For example, With n = 3, {0:[0,0,0], 1: [233, 101, 100], 2: [3, 95, 200]}
                          Class index 0 is always encoded as [0,0,0] background color.
        """

        super(Segmentation, self).__init__(model, model_dir, load_saved_model,
                                           model_file_name, use_gpu)
        assert isinstance(num_classes, int), "should be the number of classes"
        assert num_classes >= 1, "for segmentation task, it should be greater than 1 class"

        self.num_classes = num_classes
        self.threshold = threshold

        if color_map:
            assert isinstance(color_map, dict)
            self.class_index_to_color = color_map
        else:
            if self.num_classes == 1:
                self.class_index_to_color = {0: 0, 1: 255}
            else:
                self.class_index_to_color = {0: [0, 0, 0]}
                additional_colors = np.random.randint(0, 256, size=(self.num_classes - 1, 3))
                # Create random RGB color triplets
                for index, color in enumerate(additional_colors.tolist()):
                    self.class_index_to_color[index + 1] = color

        self.__create_color_palette()

    def __create_color_palette(self):
        if self.num_classes == 1:
            self.palette = [0, 0, 0, 255, 255, 255]
        else:
            self.palette = np.array(list(self.class_index_to_color.values())).flatten().astype(np.uint8).tolist()

        self.palette = self.palette + list(np.zeros(768 - (len(self.palette)), dtype=np.uint8).tolist())

    def predict_batch(self, x: Union[torch.Tensor, np.ndarray], *args, **kwargs) -> torch.Tensor:
        x = (self.move_input_to_device(x) if "non_blocking" not in kwargs
             else self.move_input_to_device(x, kwargs["non_blocking"]))

        pred = self._model(x)

        if isinstance(pred, dict) and 'out' in pred:
            return pred['out']  # torchvision model's returns prediction in OrderedDict
        else:
            return pred

    def save_prediction(self, loader: torch.utils.data.DataLoader, save_dir: str):
        """
        Accepts torch data loader, performs prediction and save prediction as png image into save directory.
        Loader must return (batch_tensor, image_file_names)

        :param loader: torch.data loaders
        :param save_dir : the output path to save predicted segmentation mask
        :return: tuple of torch.Tensor of (prediction, targets)
        """
        assert loader is not None and len(loader) > 0
        assert save_dir is not None, "Output directory should not be none."

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        self._model = self._model.to(self._device)
        self._model.eval()
        with torch.no_grad():
            for x, y in tqdm(loader, total=len(loader), desc="{:12s}".format('Prediction')):
                y_pred = self.predict_batch(x).cpu()
                output_mask = self.transform_output(y_pred)
                self._save_image_batch(output_mask, y, save_dir)

    def _save_image_batch(self, class_indices: torch.Tensor, filenames: List[str], save_dir: str):
        assert class_indices.ndim == 3, "should be in the form of BHW"
        for i in range(class_indices.shape[0]):
            image = Image.fromarray(class_indices[i].cpu().numpy().astype(np.uint8), "P")
            image.putpalette(self.palette)
            filename = filenames[i]
            if not filename.endswith(".png"):
                filename = f"{filename.split('.')[0]}.png"
            image.save(os.path.join(save_dir, filename))

    def predict_class(self, loader: torch.utils.data.DataLoader):
        raise NotImplementedError()

    def show_predictions(self, loader: torch.utils.data.DataLoader,
                         image_inverse_transform: Callable = None,
                         samples: int = 4, cols: int = 3, figsize: Tuple[int, int] = (16, 16),
                         target_known: bool = True):
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

            if self.num_classes == 1:
                target_mask = torch.cat([target_mask, target_mask, target_mask], dim=3)
                output_mask = torch.cat([output_mask, output_mask, output_mask], dim=3)

            images = []
            for i in range(x.shape[0]):
                images.extend([x[i], target_mask[i], output_mask[i]])

            image_titles = ["Input", "Target", "Prediction"] * x.shape[0]
            plot_images(images, image_titles, cols=cols, figsize=figsize, fontsize=12)

    def transform_target(self, y: torch.Tensor):
        return self.decode_segmentation_mask(y)

    def transform_output(self, predictions: torch.Tensor) -> torch.Tensor:

        assert predictions.ndim == 4  # B,C,H,W

        if predictions.shape[1] == 1:
            # Binary
            probability = torch.sigmoid(predictions).squeeze(dim=1)
            class_indices = torch.zeros_like(probability)
            class_indices[probability >= self.threshold] = 1
        else:
            # Multiclass
            probability = torch.softmax(predictions, dim=1)
            class_indices = torch.argmax(probability, dim=1)

        return class_indices

    def decode_segmentation_mask(self, class_indices: torch.Tensor):
        """ Convert segmentation mask into RGB color image.

        :param class_indices: batch of segmentation mask in #BHW format
        :return: batch of decoded RGB images in #BCHW format
        """
        assert class_indices.ndim == 3 or class_indices.ndim == 4  # BCHW or #BHW

        if class_indices.ndim == 4:
            # Need to convert #BCHW --> #BHW
            class_indices = self.transform_output(class_indices)

        decoded_images = []
        out_channel = 3 if self.num_classes > 1 else 1

        # For each image in the batch
        for i in range(class_indices.shape[0]):
            output_mask = np.zeros((*class_indices[i].shape, out_channel), dtype=np.uint8)  # H,W,C
            for label in class_indices[i].unique():
                idx = class_indices[i] == label
                output_mask[idx] = self.class_index_to_color[int(label.item())]
            decoded_images.append(torch.from_numpy(output_mask.transpose(2, 0, 1)))

        return torch.stack(decoded_images)

    def write_prediction_to_logger(self, tag: str, loader: torch.utils.data.DataLoader,
                                   logger: MLExperimentLogger,
                                   image_inverse_transform: Callable,
                                   global_step: int, img_size: Union[int, Tuple[int, int], None] = 224):
        """
        Writes Input, Target and prediction images to the tensorboard if img_size is not None
        :param tag:
        :param loader:
        :param logger:
        :param image_inverse_transform:
        :param global_step:
        :param img_size:
        :return:
        """

        if img_size is not None:
            self._model = self._model.to(self._device)
            self._model.eval()

            with torch.no_grad():
                x, targets = get_random_samples_batch_from_loader(loader)
                predictions = self.predict_batch(x).cpu()

                x = self.transform_input(x, image_inverse_transform)
                target_mask = self.decode_segmentation_mask(targets)
                class_indices = self.transform_output(predictions)
                output_mask = self.decode_segmentation_mask(class_indices)

                target_mask = target_mask.to(torch.uint8)
                output_mask = output_mask.to(torch.uint8)

                logger.log_artifact(f"{tag} Input Image", x, global_step)
                logger.log_artifact(f"{tag} Target Mask", target_mask, global_step)
                logger.log_artifact(f"{tag} Output Mask", output_mask, global_step)


class ImageRegression(NeuralNetTask):
    """
    The class useful to perform image regression.
    """

    def __init__(self, model: torch.nn.Module, model_dir: str, load_saved_model: bool = False,
                 model_file_name: str = 'latest_model.pt', use_gpu: bool = True):
        super(ImageRegression, self).__init__(model, model_dir, load_saved_model,
                                              model_file_name, use_gpu)

    def show_predictions(self, loader: torch.utils.data.DataLoader,
                         image_inverse_transform: Callable = None,
                         samples: int = 9, cols: int = 3, figsize: Tuple[int, int] = (10, 10),
                         target_known: bool = True):
        """
        Shows predictions of random samples from loader. Draws matplotlib figure.
        :param loader: the torch data loader
        :param image_inverse_transform: the inverse transform for image
        :param samples: the number of samples. Default is 9
        :param cols: the number of columns to use while displaying images. Default is 3
        :param figsize: the matplotlib figure size to use. Default is 10, 10
        :param target_known: whether target is present in the torch dataset or not.
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
                prediction = self.transform_output(prediction)
                if target_known:
                    return f'Ground Truth={self.transform_target(y)}\nPrediction={prediction}'
                else:
                    return f"{y}\nPrediction={prediction}"

            image_title_generator = ((x[index], create_title(y[index], predictions[index]),
                                      title_color)
                                     for index in range(x.shape[0]))

            plot_images_with_title(image_title_generator, samples=samples, cols=cols, figsize=figsize)

    def transform_target(self, y: torch.Tensor):
        """
        Accepts torch Tensor
        :param y:
        :return:
        """
        return round(y.item(), 2)

    def transform_output(self, prediction: torch.Tensor):
        """
        Accepts torch Tensor
        :param prediction:
        :return:
        """
        return round(prediction.item(), 2)

    def write_prediction_to_logger(self, tag: str, loader: torch.utils.data.DataLoader,
                                   logger: MLExperimentLogger,
                                   image_inverse_transform: Callable, global_step: int,
                                   img_size: Union[int, Tuple[int, int], None] = 224):
        """
        Writes prediction to TensorBoard

        :param tag: unique tag
        :param loader: the torch data loader
        :param logger: tensorboard writer object
        :param image_inverse_transform: reverse image transform
        :param global_step: the epoch value
        :param img_size: image size to use while writing image to tensorboard. Default is 224.
        :return: None
        """

        if img_size:
            assert isinstance(img_size, int) or (isinstance(img_size, tuple) and len(img_size) == 2)
        else:
            return

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

            logger.log_artifact(f'{tag}', torch.stack(output_images), global_step)

    def predict_class(self, loader):
        raise NotImplementedError()


class ImageClassification(NeuralNetTask):
    """
    The class useful for image classification task.
    """

    def __init__(self, model: torch.nn.Module, model_dir: str, load_saved_model: bool = False,
                 model_file_name: str = 'latest_model.pt', use_gpu: bool = True, classes: Sequence = None):
        super(ImageClassification, self).__init__(model, model_dir, load_saved_model,
                                                  model_file_name, use_gpu)
        self._classes = classes

    def predict_class(self, loader: torch.utils.data.DataLoader):
        predictions, targets = self.predict(loader)
        predicted_class, probability = self.transform_output(predictions)
        return predicted_class, probability, targets

    def transform_target(self, y):
        if self._classes:
            # if classes is not empty, replace target with actual class label
            y = self._classes[y]
        return y

    def transform_output(self, predictions: torch.Tensor):
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

    def _create_title_for_display(self, target_class_index, predicted_class_index, predicted_probability,
                                  target_known=True):
        predicted_class = self.transform_target(predicted_class_index)
        probability = round(predicted_probability.item(), 2)
        if target_known:
            target_class = self.transform_target(target_class_index)
            title_color = "green" if predicted_class == target_class else "red"
            return (f'Ground Truth={target_class}'
                    f'\nPrediction={predicted_class}, '
                    f'{probability}', title_color)
        else:
            return f'{target_class_index}\nPrediction={predicted_class}, {probability}', 'yellow'

    def _create_output_image_for_tensorboard(self, target_class_index, predicted_class_index, predicted_probability,
                                             img_size=(224, 224)):
        ground_truth = self.transform_target(target_class_index)
        predicted_class = self.transform_target(predicted_class_index)
        probability = round(predicted_probability.item(), 2)
        text_color = "green" if ground_truth == predicted_class else "red"
        display_content = f'{ground_truth}\n{predicted_class}, {probability}'
        return create_text_image(display_content, img_size=img_size, text_color=text_color)

    def show_predictions(self, loader: torch.utils.data.DataLoader,
                         image_inverse_transform: Callable = None, samples: int = 9, cols: int = 3,
                         figsize: Tuple[int, int] = (10, 10), target_known: bool = True):
        """
        Shows predictions of random samples from loader. Draws matplotlib figure.
        :param loader: the torch data loader
        :param image_inverse_transform: the inverse transform for image
        :param samples: the number of samples. Default is 9
        :param cols: the number of columns to use while displaying images. Default is 3
        :param figsize: the matplotlib figure size to use. Default is 10, 10
        :param target_known: whether target is present in the torch dataset or not.
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

            image_title_generator = ((x[index], *self._create_title_for_display(targets[index], class_indices[index],
                                                                                probabilities[index], target_known))
                                     for index in range(x.shape[0]))

            plot_images_with_title(image_title_generator, samples=samples, cols=cols, figsize=figsize)

    def write_prediction_to_logger(self, tag: str, loader, logger: MLExperimentLogger, image_inverse_transform,
                                   global_step: int, img_size=224):
        """
        Writes prediction to TensorBoard

        :param tag: unique tag
        :param loader: the torch data loader
        :param logger: tensorboard writer object
        :param image_inverse_transform: reverse image transform
        :param global_step: the epoch value
        :param img_size: image size to use while writing image to tensorboard. Default is 224.
        :return: None
        """

        if img_size:
            assert isinstance(img_size, int) or (isinstance(img_size, tuple) and len(img_size) == 2)
        else:
            return

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

            output_images = []
            for index in range(x.shape[0]):
                content_image = self._create_output_image_for_tensorboard(targets[index], class_indices[index],
                                                                          probabilities[index], img_size)
                if input_img_size != img_size:
                    output_images.append(to_tensor(to_pillow_image(x[index].squeeze(dim=0))))
                else:
                    output_images.append(x[index].squeeze(dim=0))

                output_images.append(to_tensor(content_image))

            logger.log_artifact(f'{tag}', torch.stack(output_images), global_step)


class MultiLabelImageClassification(ImageClassification):
    """
    The class useful for multi label image classification task.
    """

    def __init__(self, model: torch.nn.Module, model_dir, load_saved_model=False,
                 model_file_name='latest_model.pt', use_gpu=True, classes=None):
        super(MultiLabelImageClassification, self).__init__(model, model_dir, load_saved_model,
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

    def _create_title_for_display(self, target_class_indices, predicted_class_indexes, predicted_probs,
                                  target_known=True):
        predicted_classes = self.transform_target(predicted_class_indexes)
        predicted_probs = f'{", ".join([round(predicted_probs[prob_index], 2) for prob_index in predicted_class_indexes if prob_index])}'

        if target_known:
            target_classes = self.transform_target(target_class_indices)
            title_color = "green" if target_classes == predicted_classes else "red"
            return f'GT={target_classes}\nPred={predicted_classes},\n{predicted_probs}', title_color
        else:
            return f'{target_class_indices}\nPred={predicted_classes},\n{predicted_probs}', "yellow"

    def _create_output_image_for_tensorboard(self, target_class_indices, predicted_class_indexes, predicted_probs,
                                             img_size=(224, 224)):

        target_classes = self.transform_target(target_class_indices)
        predicted_classes = self.transform_target(predicted_class_indexes)
        probabilities = ", ".join(
            [round(predicted_probs[prob_index], 2) for prob_index in predicted_class_indexes if prob_index])
        display_content = f'{target_classes}\n{predicted_classes}\n{probabilities}'
        text_color = "green" if target_classes == predicted_classes else "red"
        return create_text_image(display_content, img_size=img_size, text_color=text_color)
