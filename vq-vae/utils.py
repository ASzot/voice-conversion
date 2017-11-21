import tensorflow as tf
import numpy as np

def mu_law(x, mu=255, int8=False):
  """A TF implementation of Mu-Law encoding.
  Args:
    x: The audio samples to encode.
    mu: The Mu to use in our Mu-Law.
    int8: Use int8 encoding.
  Returns:
    out: The Mu-Law encoded int8 data.
  """
  out = tf.sign(x) * tf.log(1 + mu * tf.abs(x)) / np.log(1 + mu)
  out = tf.floor(out * 128)
  if int8:
    out = tf.cast(out, tf.int8)
  return out
