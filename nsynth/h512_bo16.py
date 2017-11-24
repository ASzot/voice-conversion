# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A WaveNet-style AutoEncoder Configuration and FastGeneration Config."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
from six.moves import range  # pylint: disable=redefined-builtin
import tensorflow as tf
import utils
import masked


class Config(object):
  """Configuration object that helps manage the graph."""

  def __init__(self, train_path=None):
    self.num_iters = 200000
    self.learning_rate_schedule = {
        0: 2e-4,
        90000: 4e-4 / 3,
        120000: 6e-5,
        150000: 4e-5,
        180000: 2e-5,
        210000: 6e-6,
        240000: 2e-6,
    }
    self.ae_hop_length = 512
    self.train_path = train_path

  @staticmethod
  def _condition(x, encoding):
    """Condition the input on the encoding.

    Args:
      x: The [mb, length, channels] float tensor input.
      encoding: The [mb, encoding_length, channels] float tensor encoding.

    Returns:
      The output after broadcasting the encoding to x's shape and adding them.
    """
    mb, length, channels = x.get_shape().as_list()
    enc_mb, enc_length, enc_channels = encoding.get_shape().as_list()
    assert enc_mb == mb
    assert enc_channels == channels

    encoding = tf.reshape(encoding, [mb, enc_length, 1, channels])
    x = tf.reshape(x, [mb, enc_length, -1, channels])
    x += encoding
    x = tf.reshape(x, [mb, length, channels])
    x.set_shape([mb, length, channels])
    return x

  def build(self, inputs, K, D, beta, global_step):
    """Build the graph for this configuration.

    Args:
      inputs: A dict of inputs. For training, should contain 'wav'.
      is_training: Whether we are training or not. Not used in this config.

    Returns:
      A dict of outputs that includes the 'predictions', 'loss', the 'encoding',
      the 'quantized_input', and whatever metrics we want to track for eval.
    """
    num_stages = 10
    num_layers = 10
    filter_length = 3
    width = 512
    skip_width = 256
    ae_num_stages = 10
    ae_num_layers = 30
    ae_filter_length = 3
    ae_width = 128
    self.ae_bottleneck_width = D
    self.beta = beta
    lr = 0.001

    with tf.variable_scope('forward'):
        # Encode the source with 8-bit Mu-Law.
        #x = inputs['wav']
        x = inputs
        x_quantized = utils.mu_law(x)
        x_scaled = tf.cast(x_quantized, tf.float32) / 128.0

        with tf.variable_scope('embed') :
            embeds = tf.get_variable('embed', [K, D],
                    initializer=tf.truncated_normal_initializer(stddev=0.02))

        ###
        # The Non-Causal Temporal Encoder.
        ###

        with tf.variable_scope('enc') as enc_param_scope:
            en = masked.conv1d(
                x_scaled,
                causal=False,
                num_filters=ae_width,
                filter_length=ae_filter_length,
                name='ae_startconv')

            for num_layer in range(ae_num_layers):
              dilation = 2**(num_layer % ae_num_stages)
              d = tf.nn.relu(en)
              d = masked.conv1d(
                  d,
                  causal=False,
                  num_filters=ae_width,
                  filter_length=ae_filter_length,
                  dilation=dilation,
                  name='ae_dilatedconv_%d' % (num_layer + 1))
              d = tf.nn.relu(d)
              en += masked.conv1d(
                  d,
                  num_filters=ae_width,
                  filter_length=1,
                  name='ae_res_%d' % (num_layer + 1))

            en = masked.conv1d(
                en,
                num_filters=self.ae_bottleneck_width,
                filter_length=1,
                name='ae_bottleneck')
            en = masked.pool1d(en, self.ae_hop_length, name='ae_pool', mode='avg')

        self.enc_param_scope = enc_param_scope

        encoding = en

        z_e = encoding

        _t = tf.tile(tf.expand_dims(z_e, -2), [1, 1, K, 1]) #[batch,latent_h,latent_w,K,D]
        _e = tf.reshape(embeds, [1, 1, K, D])
        _t = tf.norm(_t - _e, axis = -1)
        k = tf.argmin(_t, axis = -1) # -> [latent_h,latent_w]
        self.k = k
        z_q = tf.gather(embeds, k)

        self.z_e = z_e
        self.k = k
        self.z_q = z_q

        en = self.z_q

        ###
        # The WaveNet Decoder.
        ###
        with tf.variable_scope('dec') as dec_param_scope:
            l = masked.shift_right(x_scaled)
            l = masked.conv1d(
                l, num_filters=width, filter_length=filter_length, name='startconv')

            # Set up skip connections.
            s = masked.conv1d(
                l, num_filters=skip_width, filter_length=1, name='skip_start')

            # Residual blocks with skip connections.
            for i in range(num_layers):
              dilation = 2**(i % num_stages)
              d = masked.conv1d(
                  l,
                  num_filters=2 * width,
                  filter_length=filter_length,
                  dilation=dilation,
                  name='dilatedconv_%d' % (i + 1))
              d = self._condition(d,
                                  masked.conv1d(
                                      en,
                                      num_filters=2 * width,
                                      filter_length=1,
                                      name='cond_map_%d' % (i + 1)))

              assert d.get_shape().as_list()[2] % 2 == 0
              m = d.get_shape().as_list()[2] // 2
              d_sigmoid = tf.sigmoid(d[:, :, :m])
              d_tanh = tf.tanh(d[:, :, m:])
              d = d_sigmoid * d_tanh

              l += masked.conv1d(
                  d, num_filters=width, filter_length=1, name='res_%d' % (i + 1))
              s += masked.conv1d(
                  d, num_filters=skip_width, filter_length=1, name='skip_%d' % (i + 1))

            s = tf.nn.relu(s)
            s = masked.conv1d(s, num_filters=skip_width, filter_length=1, name='out1')
            s = self._condition(s,
                                masked.conv1d(
                                    en,
                                    num_filters=skip_width,
                                    filter_length=1,
                                    name='cond_map_out1'))
            s = tf.nn.relu(s)

        self.dec_param_scope = dec_param_scope

        ###
        # Compute the logits and get the loss.
        ###
        logits = masked.conv1d(s, num_filters=256, filter_length=1, name='logits')
        logits = tf.reshape(logits, [-1, 256])
        probs = tf.nn.softmax(logits, name='softmax')
        x_indices = tf.cast(tf.reshape(x_quantized, [-1]), tf.int32) + 128

        self.recon = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=x_indices, name='nll'),
            0,
            name='recon_loss')

        self.vq = tf.reduce_mean(
                    tf.norm(tf.stop_gradient(self.z_e) - z_q,axis=-1)**2,
                    axis=[0,1])
        self.commit = tf.reduce_mean(
            tf.norm(self.z_e - tf.stop_gradient(z_q),axis=-1)**2,
            axis=[0,1])
        self.loss = self.recon + self.vq + beta * self.commit

    with tf.variable_scope('backward'):
      # Decoder grads
      decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.dec_param_scope.name)
      decoder_grads = list(zip(tf.gradients(self.loss,decoder_vars),decoder_vars))

      # Encoder Grads
      encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.enc_param_scope.name)
      grad_z = tf.gradients(self.recon,self.z_q)
      encoder_grads = [(tf.gradients(self.z_e,var,grad_z)[0]+self.beta*tf.gradients(self.commit,var)[0],var) for var in encoder_vars]

      # Embedding Grads
      embed_grads = list(zip(tf.gradients(self.vq,embeds),[embeds]))

      optimizer = tf.train.AdamOptimizer(lr)
      self.train_op= optimizer.apply_gradients(decoder_grads+encoder_grads+embed_grads,global_step=global_step)

    return {
        'predictions': probs,
        'loss': self.loss,
        'train_op': self.train_op,
        'eval': {
            'nll': self.loss
        },
        'quantized_input': x_quantized,
        'encoding': encoding,
    }

