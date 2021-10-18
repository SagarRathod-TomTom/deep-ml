# deep-ml


This library is a wrapper around pytorch and useful for solving image classification and semantic
segmentation problems.

### Features
1. Easy to use wrapper around pytorch so that you can focus on training and
   validating your model.

2. Integrates with Tensorboard to use it to monitor metrics while model trains.

3. Quickly visualize your model's predictions.

4. Following are different types of machine learning tasks available to choose from:

   * ImageClassification
   * MultiLabelImageClassification
   * ImageRegression
   * Segmentation

# Installation

Before installing **deepml**, it is recommended to refer [pytorch](https://pytorch.org/) official page for **torch** installation.

### Pypi
```bash
pip install deepml
```

# Usage

#### 1. Create torch data loaders.
```python
import torch

train_loader = # your train loader instance of torch.utils.data.DataLoader
val_loader = # your val loader instance of torch.utils.data.DataLoader
```

#### 2. Create your deep neural net architecture.

```python
import torchvision

# instance of torch.nn.Module
model = torchvision.models.vgg.vgg19(pretrained=False)
```

#### 3. Choose your machine learning task.

```python
from deepml.tasks import ImageClassification
classification = ImageClassification(model, model_dir="experiment1", 
                                     load_saved_model=False, 
                                     classes=['class1', 'class2', 'class3'])
```

#### 4. Define optimizer, loss function and lr scheduler.

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# loss function
criterion = torch.nn.CrossEntropyLoss()

# Choose lr_scheduler if any
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
```

#### 5. Quickly start training your model using deepml.train.Learner class.

```python
from deepml.train import Learner
# instantiate learner class
learner = Learner(classification, optimizer, criterion)

# Fit Learner
learner.fit(train_loader, val_loader, epochs=10, lr_scheduler=lr_scheduler)
```

##### 6. Use tensorboard to visualize model loss and metrics.

##### On Google Colab or Jupyter Notebook:

```bash
%load_ext tensorboard
%tensorboard --logdir 'experiment1'
```
##### On OS:
```bash
tensorboard --logdir 'experiment1'
```

#### 7. Quickly see some samples predictions from data loader.
```python
learner.show_predictions(val_loader, samples=30, cols=6, figsize=(20, 20))
```

#### 8. Run prediction on data loader.
```python
predictions, targets = learner.predict(val_loader)
```

# Examples
Check out the below google colaboratory notebook examples:

1. [Image Regression](https://colab.research.google.com/github/sagar-rathod/PytorchDeepML/blob/master/examples/Image_Regression_Example.ipynb)
2. [Image Classification](https://colab.research.google.com/github/sagar-rathod/PytorchDeepML/blob/master/examples/Image_Classification_Example.ipynb)
3. [Binary Semantic Segmentation]()
4. [Multiclass Semantic Segmentation]()


# Contributing
deepml is an open source project and anyone is welcome to contribute. An easy way to get started is by suggesting a new enhancement on the Issues. If you have found a bug, then either report this through Issues, or even better, make a fork of the repository, fix the bug and then create a Pull Request to get the fix into the master branch.


# License
deepml is available under the MIT License. For details see the [LICENSE](https://github.com/sagar-rathod/PytorchDeepML/blob/master/LICENSE) file.

