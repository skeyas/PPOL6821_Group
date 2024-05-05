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

```
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
### Images sizes in CNNs
With Convolutional Neural Networks, we need to pay attention to the compatibility between the input data shape and the configuration of the input layer of the network, essentially ensuring that the size of the input data matches the expected input size of the network.

If you attempt to train or evaluate a model using images of a different size than the one it expects, Keras won't raise an error during model compilation. However, during training or inference, the images will be read and processed according to the specified input shape. If the actual input images are larger than the expected size, it will automatically crop or resize them to fit. Conversely, if the input images are smaller, the model will only read a portion of the images commensurate with the shape it expects, leading to unexpected results and.or unstable training. 

A quick way to validate it:
```python

# assuming cnn has been compiled prior and is called is called model
sample_batch = next(iter(train_dataset))
if sample_batch[0].shape[1:] != model.input_shape[1:]:
    raise ValueError(f"Input data dimensions are {sample_batch[0].shape[1:]}, but the model expects {model.input_shape[1:}.")
```
### Dimensionality issues in [Transformers for images](https://keras.io/examples/vision/swin_transformers/) 
#### (a) Image Dimension and Patch Size Compatibility
One common issue arises when the image dimensions are not divisible by the patch size. This is important to keep in mind because the architecture processes images by dividing them into patches, multiple times and if the dimensions of the image are not perfectly divisible by the patch dimensions, this will lead to errors or the need for padding, which can affect performance and results.  
  
Generally images sizes in the power of 2 with a patch embedding size of (2,2) works well especially because later in the patch merging operation, four distinct slices are extracted, reshaped and concated. Each extraction step relies on being able to access patches in a grid that is evenly divisible. If the dimensions are not divisible by 2, some patches at the edges might not have corresponding neighbors to form a complete group of four, leading to issues in merging.

#### (b) Embedding Dimension and Number of Heads
The embedding dimension should be divisible by the number of heads in the multi-head attention mechanism. This is important for the model to evenly distribute the embedding vector across different heads. A mismatch here can lead to runtime errors or inefficient computation. Adjust the embedding dimension to be a multiple of the number of attention heads. For instance, with 8 heads, good embedding dimensions could be 64, 128, 256, etc. 

### Hitches of Migrating Keras 2 to Keras 3

#### Deprecated backend module and the new Ops API 
`keras.backend` in Keras 2 serves as an abstraction layer, allowing develpers to write code that can raun on multiple deep learning frameworks. In practice, it is used to conduct basic math operations, tensor manipulations, linear algebra operations, etc. 

Here are some examples:
```python
import keras.backend as K

K.sum() # sum of all elements in a tensor
K.mean() # mean of all elements in a tensor
K.clip() # clip tensor values to a specified range
K.round() # round tensor values to the nearest integer
K.dot() # dot product of two tensors
K.log() # natural logarithm of tensor values
k.epsilon() # a small constant value to avoid division by zero
```
`keras.backend` is deprecated in Keras 3, and the more powerful [Ops API](https://keras.io/api/ops/) is introduced. The Keras 3 Ops API provides a comprehensive set of operations and functions that extend the capabilities of what can be achieved within the Keras framework. Categories of operations in Keras 3 Ops API include: NumPy Ops, NN Ops, Linear Algebra Ops, Core Ops, Image Ops, and FFT Opes. 

For some other operations, the deprecated `keras.backend` functions can be replaced with tf.math functions.

Using the same examples as above:

```python
import tensorflow as tf
import keras

# K.sum() is replaced with
tf.math.reduce_sum()
# K.mean() is replaced with
tf.math.reduce_mean()
# K.clip() is replaced with
keras.ops.clip() # or
tf.clip_by_value()
# K.round() is replaced with
keras.ops.round() # or
tf.round()
# K.dot() is replaced with
keras.ops.dot() # or
tf.tensordot()
# K.log() is replaced with
keras.ops.log() # and other variants such as log10(), log1p(), log2(). or
tf.math.log()
```

#### F1Score
There is no F1Score class in Keras 2. One way to calculate F1 score is to use the following code:
```python
# Define metrics: precision, recall, F1_score
import Keras
from keras import backend as K

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2*((p*r)/(p+r+K.epsilon()))
```
The deprecation of `keras.backend` in Keras 3 means that the above code will not work. The good news is that the [F1Score class](https://keras.io/api/metrics/classification_metrics/#f1score-class) is introduced in Keras 3, which can be used to calculate the F1 score. Here is an example of how to use the F1Score class:

```python
from keras.metrics import F1Score

metric = F1Score(threshold=0.5)
y_true = np.array([[1,1,1],
                   [1,0,0],
                   [1,1,0],
                   [0,1,0]],
                   np.int32)
y_pred = np.array([[0.0,0.6,0.7],
                   [0.2,0.6,0.6],
                   [0.9,0.8,0.1],
                   [0.2,0.8,0.1]],
                   np.float32)
metric.update_state(y_true, y_pred)
result = metric.result()
print(result.numpy())  ## [0.49999997 0.8571428  0.66666657]
```    
Please note that `threshold` must be explicitly set when creating the F1Score object. Using `F1Score()` without setting threshold will yield wrong results. The threshold is used to convert the predicted probabilities to binary values, and the default value is 0.5.
