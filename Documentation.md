# Lessons Learned
Collective Guide for PPOL 6821: Applied Neural Nets

***

### Environment Setup
#### Virtual environments
When working locally, it is typically wise to use virtual environments,
as it reduces the risk of conflicting packages and dependencies, as well as
making it easier to recreate your solution on a different computer.

This can be achieved quickly using the following commands (Linux or MacOS):
```commandline
python -m venv venv
source venv/bin/activate
```

The source command will execute the activate script, so following pip 
installs will apply to the virtual environment, rather than the base. 
Creating a virtual environment with venv also generates a PowerShell script
and a bat script, which can be used to activate the environment in Windows
based systems.

It is also possible to use a virtual environment on Google Colab. In most
use cases, this will be unnecessary. However, if you are using deep 
learning for some specific use cases, such as reinforcement learning,
which frequently involve installing other packages, such as gymnasium or 
mujoco, a virtual environment can be helpful, as such packages involve
complex dependencies that pip cannot automatically resolve, making 
recreation extremely challenging.

### Debugging
#### Dependencies and Versions
When the documentation for a package contains code that raises an error,
a first step should be to upgrade the version of the package.
This is a frequent issue with the version of Keras installed by default
on Google Colab, noticeable in such cases as trying to use the ops module.
As of April 2024, Colab still uses version 2 of Keras by default, rather 
than version 3. Updating this is as simple as running the following 
command:

```commandline
pip install --upgrade [package]
```

To view the version of a package you have installed, you can run 
`pip show [package]`. This returns useful information such as the name
of the package, the version, where it's installed, and any dependencies 
that either it has or another package has on it. For some packages, 
such as numpy, which returns the entire license, rather than a link to 
it, `pip show` is likely more information than you need or want. If you 
just want a limited amount of information, such as to find what packages
are installed and which version, you can instead use `pip list`.

However, it is still on occasion valuable to run a pip show on a package.
If you have multiple environments or versions of Python installed on a 
system, you may be using the wrong one, and therefore would need to 
run pip show on your packages to ensure you're using the version 
associated with the correct environment and Python version.

#### Speed
When working with PyTorch, you may encounter your code taking longer than
you would expect it to take. Verify you are **setting the device**. PyTorch
supports the use of both CPUs and GPUs, and runs on the CPU by default.
To switch to the GPU, set the device to "cuda" for **both the model and the
data**, eg:

```python
model = model.to("cuda")
data = data.to("cuda")
```

#### Excess RAM usage on Colab
Sometimes when using PyTorch on Colab, you may encounter memory issues with
the GPU, where nothing is running, but the RAM is full. The easiest way to 
clear this is to restart the runtime, which is obviously inconvenient. An
alternative is to free the memory, by setting the model to None or using
`del model`, and then running `torch.cuda.empty_cache()`.

### Overfitting
Problem: The model performs well on training data but poorly on unseen data.\
Solution: Implement regularization techniques, dropout, or increase data.

```python
model = Sequential([
    Dense(128, activation="relu", input_shape=(input_shape,), kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(num_classes, activation="softmax")
])
```

### Vanishing Gradients
Problem: Weights update becomes very small, effectively stopping the network from learning in deep networks.\
Solution: Use ReLU or its variants as activation functions, initialize weights carefully.

```python
model = Sequential([
    Dense(128, activation="relu", input_shape=(input_shape,)),
    Dense(num_classes, activation="softmax")
])
```

### Learning Rate Issues
Problem: Too high a learning rate can cause the model to converge too fast to a suboptimal solution, and too low can slow down the learning process.\
Solution: Use an adaptive learning rate optimizer like Adam.

```python
model = Sequential([
    Dense(128, activation="relu", input_shape=(input_shape,)),
    Dense(num_classes, activation="softmax")
])
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
```

### Improper Filter Size
Problem: Choosing the wrong size of the filters can lead to poor feature extraction.\
Solution: Experiment with different sizes of filters to find the optimal configuration for specific tasks. Small filters (3x3) are generally preferred for capturing fine details.

```python
Conv2D(64, (3, 3), activation="relu")
```

### Insufficient Training Data
Problem: Not having enough data to train a robust model.\
Solution: Data Augmentation could increase the diversity of data available for training models.

```python
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
```
