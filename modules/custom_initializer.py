import numpy as np
import tensorflow as tf
from tensorflow.keras.initializers import Initializer
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import backend

ones_zeros = np.empty((1,512))
ones_zeros[:,::2] = 1
ones_zeros[:,1::2] = 0
ones_zeros = tf.convert_to_tensor(ones_zeros, dtype=tf.float32)

# attention_model.get_layer('attention_layer').set_weights([ones_zeros,])
# attention_model.get_layer('attention_layer').get_weights()[0]

def _get_dtype(dtype):
  if dtype is None:
    dtype = backend.floatx()
  return dtypes.as_dtype(dtype)

class custom_onezero_initializer(Initializer):
  """Initializer that generates tensors with constant values.

  Also available via the shortcut function `tf.keras.initializers.constant`.

  Only scalar values are allowed.
  The constant value provided must be convertible to the dtype requested
  when calling the initializer.

  Examples:

  >>> # Standalone usage:
  >>> initializer = tf.keras.initializers.Constant(3.)
  >>> values = initializer(shape=(2, 2))

  >>> # Usage in a Keras layer:
  >>> initializer = tf.keras.initializers.Constant(3.)
  >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)

  Args:
    value: A Python scalar.
  """

  def __init__(self, value=0):
    self.value = value

  def __call__(self, shape, dtype=None, **kwargs):
    """Returns a tensor object initialized to `self.value`.

    Args:
      shape: Shape of the tensor.
      dtype: Optional dtype of the tensor. If not specified,
       `tf.keras.backend.floatx()` is used,
       which default to `float32` unless you configured it otherwise
       (via `tf.keras.backend.set_floatx(float_dtype)`).
      **kwargs: Additional keyword arguments.
    """
    del kwargs
    ones_zeros = np.empty((1,512))
    ones_zeros[:,::2] = 1
    ones_zeros[:,1::2] = 0
    ones_zeros = tf.convert_to_tensor(ones_zeros, dtype=dtype)
    return ones_zeros

  def get_config(self):
    return {'value': self.value}