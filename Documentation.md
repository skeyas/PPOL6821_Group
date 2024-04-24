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

It is also possible to use a virtual environment on Google Colab.

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
To switch to the GPU, set the device to "cuda" for both the model and the
data, eg:
```python
model = model.to(device)
data = data.to(device)
```
#### Excess RAM usage on Colab
Sometimes when using PyTorch on Colab, you may encounter memory issues with
the GPU, where nothing is running, but the RAM is full. The easiest way to 
clear this is to restart the runtime, which is obviously inconvenient. An
alternative is to free the memory, by setting the model to None, and then 
running `torch.cuda.empty_cache()`.