import numpy as np
# from dezero.core import Variable

gpu_enable = True
try:
    import cupy as cp 
    cupy = cp
except ImportError:
    gpu_enable = False


def get_array_module(x):
    """Returns the array module for `x`.

    Args:
        x (dezero.Variable or numpy.ndarray or cupy.ndarray): Values to
            determine whether NumPy or CuPy should be used.

    Returns:
        module: `cupy` or `numpy` is returned based on the argument.
    """
    from dezero.core import Variable
    if isinstance(x, Variable):
        x = x.data

    if not gpu_enable:
        return np
    xp = cp.get_array_module(x)
    return xp


def as_numpy(x):
    if isinstance(x, Variable):
        x = x.data

    if np.isscalar(x):
        return np.array(x)
    elif isinstance(x, np.ndarray):
        return x
    return cp.asnumpy(x)

def as_cupy(x):
    if isinstance(x, Variable):
        x = x.data

    if not gpu_enable:
        raise Exception('Cannot load CuPy. You need to install CuPy!')
    return cp.asarray(x)

