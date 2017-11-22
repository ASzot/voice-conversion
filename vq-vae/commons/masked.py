from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def conv1d(x,
           num_filters,
           filter_length,
           name,
           dilation=1,
           causal=True,
           kernel_initializer=tf.uniform_unit_scaling_initializer(1.0),
           biases_initializer=tf.constant_initializer(0.0)):
  """Fast 1D convolution that supports causal padding and dilation.
  Args:
    x: The [mb, time, channels] float tensor that we convolve.
    num_filters: The number of filter maps in the convolution.
    filter_length: The integer length of the filter.
    name: The name of the scope for the variables.
    dilation: The amount of dilation.
    causal: Whether or not this is a causal convolution.
    kernel_initializer: The kernel initialization function.
    biases_initializer: The biases initialization function.
  Returns:
    y: The output of the 1D convolution.
  """
  batch_size, length, num_input_channels = x.get_shape().as_list()
  assert length % dilation == 0

  kernel_shape = [1, filter_length, num_input_channels, num_filters]
  strides = [1, 1, 1, 1]
  biases_shape = [num_filters]
  padding = 'VALID' if causal else 'SAME'

  with tf.variable_scope(name):
    weights = tf.get_variable(
        'W', shape=kernel_shape, initializer=kernel_initializer)
    biases = tf.get_variable(
        'biases', shape=biases_shape, initializer=biases_initializer)

  x_ttb = time_to_batch(x, dilation)
  if filter_length > 1 and causal:
    x_ttb = tf.pad(x_ttb, [[0, 0], [filter_length - 1, 0], [0, 0]])

  x_ttb_shape = x_ttb.get_shape().as_list()
  x_4d = tf.reshape(x_ttb, [x_ttb_shape[0], 1,
                            x_ttb_shape[1], num_input_channels])
  y = tf.nn.conv2d(x_4d, weights, strides, padding=padding)
  y = tf.nn.bias_add(y, biases)
  y_shape = y.get_shape().as_list()
  y = tf.reshape(y, [y_shape[0], y_shape[2], num_filters])
  y = batch_to_time(y, dilation)
  y.set_shape([batch_size, length, num_filters])
  return y


def shift_right(x):
  """Shift the input over by one and a zero to the front.
  Args:
    x: The [mb, time, channels] tensor input.
  Returns:
    x_sliced: The [mb, time, channels] tensor output.
  """
  shape = x.get_shape().as_list()
  x_padded = tf.pad(x, [[0, 0], [1, 0], [0, 0]])
  x_sliced = tf.slice(x_padded, [0, 0, 0], tf.stack([-1, shape[1], -1]))
  x_sliced.set_shape(shape)
  return x_sliced