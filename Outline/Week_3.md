# Week 3
By now all of you should have a working dataloader, now it is time to train your network.

## Best practices for Training a Neural Network

### Datasplit
It is important that before you do anything you perform a three-way data split. A typical ration of dividing the data is:
* (70%) Training data
* (15%) Validation data
* (15%) Test data

The test data should be set aside until the very end of the project, where you use it once to test your final network.
This is done to assure that after training our dataset on the train data, and (possibly manually) tuning the hyperparameters (e.g. batch-size, learning rate, network architecture, ...) using the validation set, that the network <mark> generalises well to unseen/new data <mark>

### How to Prevent Overfitting

The basic idea behind **overfitting** is that the model is fitting too closely to the training data, and it is learning the noise or random variations in the data. As a result, it is not able to generalize well to new data, and it performs poorly on the test set or in real-world applications.

Overfitting can be detected by monitoring the learning curves, which are the plots of the training and validation loss and accuracy as a function of the number of iterations. If the training loss and accuracy continue to decrease while the validation loss and accuracy start to plateau, the model is likely to be overfitting.

Preventing overfitting in image classification tasks can be achieved by using several techniques, some of them include:

1. **Data augmentation**: Ideally we would like a lot more data, but if you are not working at a big tech company that can make half the planet label captchas for your then the next best thing is data augmentation. Data augmentation techniques, such as rotation, flipping, and cropping, can be used to increase the number of training examples and to reduce overfitting.
2. **Dropout**: Dropout is a regularization technique that randomly drops out neurons during training, this helps to reduce overfitting by preventing the model from relying too heavily on any one feature.
3. **Early stopping**: Early stopping can be used to stop the training process when the performance of the model on the validation set starts to degrade.
4. **Regularization**: Regularization methods, such as L1 and L2 regularization, can be used to prevent overfitting by adding a penalty term to the loss function that discourages the model from having too many parameters.
5. **Batch normalization**: Batch normalization can help to reduce overfitting by normalizing the activations of the neurons and reducing the internal covariate shift.
6. **Transfer learning**: Transfer learning is a technique that allows you to use a pre-trained model as a starting point and fine-tune it on your own dataset. This can be useful for image classification tasks where the amount of data is limited.
7. **Using smaller models**: using smaller models can help to prevent overfitting by reducing the number of parameters and the complexity of the model.


### Hyperparameter Search
Finding good hyperparameters is an art init of its own. I strongly advice skimming the literature close to your task at hand to get an idea where to start looking.
Some generic algorithms include:
1. Grid search: A technique where a predefined set of hyperparameters are specified and the model is trained and evaluated for each combination of them.
2. Random search: A technique where random values for the hyperparameters are selected and the model is trained and evaluated for each set of values.


## Tools to Checkout
* **torchinfo** <br>
    torchinfo is a Python library for analyzing PyTorch models and tensors. It provides a set of functions that can be used to inspect the structure and properties of PyTorch models and tensors, such as the number of layers, the number of parameters, the memory usage, and the data type. Especially useful for debuggin issues relating the shapes of your inputs.
    ```python
    from torchinfo import summary
    model = MyPytorchModel()
    batch_size = 128
    summary(model, input_size=(batch_size, 1, 96, 96))
    ```
    ```
    ==========================================================================================
    Layer (type:depth-idx)                   Output Shape              Param #
    ==========================================================================================
    KeypointModel                            [128, 30]                 --
    ├─Sequential: 1-1                        [128, 30]                 --
    │    └─Conv2d: 2-1                       [128, 32, 93, 93]         544
    │    └─ELU: 2-2                          [128, 32, 93, 93]         --
    │    └─MaxPool2d: 2-3                    [128, 32, 46, 46]         --
    │    └─Dropout: 2-4                      [128, 32, 46, 46]         --
    │    └─Conv2d: 2-5                       [128, 64, 44, 44]         18,496
    │    └─ELU: 2-6                          [128, 64, 44, 44]         --
    │    └─MaxPool2d: 2-7                    [128, 64, 22, 22]         --
    │    └─Dropout: 2-8                      [128, 64, 22, 22]         --
    │    └─Conv2d: 2-9                       [128, 128, 21, 21]        32,896
    │    └─ELU: 2-10                         [128, 128, 21, 21]        --
    │    └─MaxPool2d: 2-11                   [128, 128, 10, 10]        --
    │    └─Dropout: 2-12                     [128, 128, 10, 10]        --
    │    └─Conv2d: 2-13                      [128, 256, 10, 10]        33,024
    │    └─ELU: 2-14                         [128, 256, 10, 10]        --
    │    └─MaxPool2d: 2-15                   [128, 256, 5, 5]          --
    │    └─Dropout: 2-16                     [128, 256, 5, 5]          --
    │    └─Flatten: 2-17                     [128, 6400]               --
    │    └─Linear: 2-18                      [128, 512]                3,277,312
    │    └─ELU: 2-19                         [128, 512]                --
    │    └─Dropout: 2-20                     [128, 512]                --
    ...
    Input size (MB): 4.72
    Forward/backward pass size (MB): 495.38
    Params size (MB): 14.56
    Estimated Total Size (MB): 514.66
    =========================================================================================
    ```
* **TensorBoard** <br>
    a visualization tool provided with TensorFlow that allows you to interactively visualize and inspect the training and performance of machine learning models.

* **Pytorch Lightning** <br>
    a lightweight wrapper on top of the PyTorch library that makes it easier to train and debug machine learning models. It provides a high-level interface for training and evaluating models, while still allowing you to use the low-level PyTorch API when needed.

    Some of the key features of PyTorch Lightning include:

    1. Modular structure: PyTorch Lightning allows you to organize your code in a modular way, by separating the logic for the model, data loading, and training. This makes it easier to write, understand, and reuse code.

    2. Automatic logging: PyTorch Lightning automatically logs the training and validation metrics, as well as the model structure and the optimizer state. This makes it easy to visualize and debug the training process. -> TensorBoard

    3. Multi-GPU support: PyTorch Lightning makes it easy to train models on multiple GPUs, and it also supports distributed training.

* **Optuna** <br>
    Optuna is an open-source library for hyperparameter tuning and optimization. It allows you to define a search space for the hyperparameters of your model and then uses different optimization algorithms to find the best set of hyperparameters that minimize the loss function.

    Optuna can be used with PyTorch by integrating it with the PyTorch training loop. This can be done by defining a PyTorch optimizer and a PyTorch training loop and then wrapping the optimizer and the training loop with Optuna's study object. Optuna will then use the study object to sample the hyperparameters and evaluate the performance of the model.

    Here's an example of how you can use Optuna with PyTorch:
    ```python
    import optuna
    import torch

    def train_loop(lr: float, weight_decay: float, n_epoch = 10):
        # Define the PyTorch model, optimizer and loss function
        model = ...
        optimizer = torch.optim.Adam(model.parameters(), lr=lr weight_decay=weight_decay)
        criterion = ...
        # Define the training loop
        for epoch in range(n_epochs):
            ...
        return loss

    def objective(trial):
        # Define the search space for the hyperparameters
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
        weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)
        ...
        loss = train_loop(lr, weight_decay)
        return loss

    study = optuna.create_study()
    study.optimize(objective, n_trials=100)
    ```
    In this example, Optuna is used to sample the values for the learning rate and weight decay hyperparameters, and then it is used to evaluate the performance of the model by running the training loop with these hyperparameters. The study object is used to control the optimization process and to record the results of the trials.

    Further it allows you to parallelize hyperparameter searches over multiple threads or processes without modifying code. This makes sense if you have a lot of GPU workpower at hand and want to speed up the process.