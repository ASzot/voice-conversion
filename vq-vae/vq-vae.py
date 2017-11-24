from six.moves import xrange
import better_exceptions
import tensorflow as tf
from commons import masked
import numpy as np
from commons.ops import *
import os
import time
import json
from utils import mu_law

from wavenet.audio_reader import AudioReader
from wavenet.model import WaveNetModel

def _audio_arch(d):
    with tf.variable_scope('enc') as enc_param_scope:
        enc_spec = [
            Conv1d('conv1d_1', 1, d, k_w=4, d_w=2),
            lambda t,**kwargs : tf.nn.relu(t),
            Conv1d('conv1d_2', d, d, k_w=4, d_w=2),
            lambda t,**kwargs : tf.nn.relu(t),
            Conv1d('conv1d_3', d, d, k_w=4, d_w=2),
            lambda t,**kwargs : tf.nn.relu(t),
            Conv1d('conv1d_4', d, d, k_w=4, d_w=2),
            lambda t,**kwargs : tf.nn.relu(t),
            Conv1d('conv1d_5', d, d, k_w=4, d_w=2),
            lambda t,**kwargs : tf.nn.relu(t),
            Conv1d('conv1d_6', d, d, k_w=4, d_w=2),
        ]

    return enc_spec, enc_param_scope, None, None

class VQVAE():
    """ Class for VQ-VAE architecture

    Parameters:
        lr (float)  learning rate.
        x (batch_size, time(?), ?) input tensor.
        K (int)     Number of embeddings.
        D (int)     Dimension of embedding.

    """
    def __init__(self, lr, global_step, beta,
                 x,K,D,
                 arch_fn,
                 sess,
                 param_scope,
                 is_training=False):
        with tf.variable_scope(param_scope):
            enc_spec, enc_param_scope, dec_spec, dec_param_scope = arch_fn(D)
            with tf.variable_scope('embed') :
                embeds = tf.get_variable('embed', [K, D],
                                        initializer=tf.truncated_normal_initializer(stddev=0.02))

        with tf.variable_scope('forward') as forward_scope:
            # Encoder Pass
            x_quantized = mu_law(x)
            x_scaled = tf.cast(x_quantized, tf.float32) / 128.0
            # Why are we not expanding dim here? b/c we are defaulting to batch size of 1?
            #self.x_scaled = x_scaled

            self.input_mean = tf.reduce_mean(x_scaled)

            _t = x_scaled
            for block in enc_spec :
                _t = block(_t)
            z_e = _t

            self.z_e_mean = tf.reduce_mean(z_e)

            # Middle Area (Compression or Discretize)
            # TODO: Gross.. use brodcast instead!

            _t = tf.tile(tf.expand_dims(z_e, -2), [1, 1, K, 1]) #[batch,latent_h,latent_w,K,D]
            _e = tf.reshape(embeds, [1, 1, K, D])
            _t = tf.norm(_t - _e, axis = -1)
            k = tf.argmin(_t, axis = -1) # -> [latent_h,latent_w]
            self.k = k
            z_q = tf.gather(embeds, k)

            self.z_q_mean = tf.reduce_mean(z_q)

            self.z_e = z_e # -> [batch,latent_h,latent_w,D]
            self.k = k
            self.z_q = z_q # -> [batch,latent_h,latent_w,D]

            # End early
            #return

            # Decoder Pass
            # Copy just to be safe...?
            _t = tf.identity(z_q)

            # THINGS TO DO
            # 1. check if x is right dimension, no need to expand dim?
            # 2. check if s is same dim as x
            # 3. add conditional on speaker id (can do after reconstruction)

            num_stages = 10 # Has to do with dilation stages
            num_layers = 1 # Could lower the amount of layers
            filter_length = 3
            width = 512
            skip_width = 256

            with tf.variable_scope('dec') as dec_param_scope:
                # May need to have x be an expanded dim
                l = masked.shift_right(x_scaled)

                self.l0 = tf.reduce_mean(tf.identity(l))

                l, W_mean, b_mean = masked.conv1d_log(l, num_filters=width, filter_length=filter_length, name='startconv_dec')
                self.W_mean = W_mean
                self.b_mean = b_mean

                self.l1 = tf.reduce_mean(tf.identity(l))

                # Skip connection
                s = masked.conv1d(l, num_filters=skip_width, filter_length=1, name='skip_start_dec')

                self.s0 = tf.reduce_mean(tf.identity(s))

                # Residual blocks with skip connection
                for i in xrange(num_layers):
                    dilation = 2 ** (i % num_stages)
                    d = masked.conv1d(l, num_filters = 2 * width, filter_length = filter_length,
                        dilation = dilation, name = 'dilatedconv_%d' % (i+1))

                    self.d0 = tf.reduce_mean(tf.identity(d))

                    # Condition on z_q
                    d = self._condition(d, masked.conv1d(_t, num_filters=2*width, filter_length=1, name='cond_map_%d' % (i+1)))
                    self.d1 = tf.reduce_mean(tf.identity(d))

                    assert d.get_shape().as_list()[2] % 2 == 0

                    m = d.get_shape().as_list()[2] // 2
                    d_sigmoid = tf.sigmoid(d[:, :, :m])
                    d_tanh = tf.tanh(d[:, :, m:])
                    d = d_sigmoid * d_tanh

                    self.d2 = tf.reduce_mean(tf.identity(d))

                    l += masked.conv1d(d, num_filters=width, filter_length=1, name='res_%d' % (i+1))
                    self.l2 = tf.reduce_mean(tf.identity(l))

                    s += masked.conv1d(d, num_filters=skip_width, filter_length=1, name='skip_%d' % (i+1))
                    self.s1 = tf.reduce_mean(tf.identity(s))

                s = tf.nn.relu(s)
                s = masked.conv1d(s, num_filters=skip_width, filter_length=1, name='out1')
                self.s2 = tf.reduce_mean(tf.identity(s))

                # Condition on z_q again.
                s = self._condition(s, masked.conv1d(_t, num_filters=skip_width, filter_length=1, name='cond_map_out1'))
                s = tf.nn.relu(s)
                self.s3 = tf.reduce_mean(tf.identity(s))

                # Should this parameter be trainable...?
                logits = masked.conv1d(s, num_filters=256, filter_length=1, name='logits')
                self.logits_mean = tf.reduce_mean(tf.identity(logits))

            # Losses
            # CHECK AXES FOR REDUCE MEAN ON RECON LOSS
            logits = tf.reshape(logits, [-1, 256])

            #probs = tf.nn.softmax(logits, name='softmax')
            x_indices = tf.cast(tf.reshape(x_quantized, [-1]), tf.int32) + 128
            self.indices_mean = tf.reduce_mean(x_indices)

            self.recon = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=logits, labels=x_indices, name='nll'),
                    0, name='recon_loss')

            # Reconstruction loss for images.
            #self.recon = tf.reduce_mean((self.p_x_z - x) ** 2, axis=[0,1,2,3])

            sg_e = tf.stop_gradient(self.z_e)
            sg_norm = tf.norm(sg_e - z_q, axis=-1) ** 2

            self.vq = tf.reduce_mean(sg_norm, axis=[0,1])
            self.commit = tf.reduce_mean(
                tf.norm(self.z_e - tf.stop_gradient(z_q), axis=-1) ** 2,
                axis=[0,1])
            self.loss = self.recon + self.vq + beta * self.commit

            # NLL
            # TODO: is it correct impl?
            # it seems tf.reduce_prod(tf.shape(self.z_q)[1:2]) should be multipled
            # in front of log(1/K) if we assume uniform prior on z.
            #self.nll = -1.*(tf.reduce_mean(tf.log(self.p_x_z),axis=[1,2]) + tf.log(1/tf.cast(K,tf.float32)))/tf.log(2.)

        if( is_training ):
            with tf.variable_scope('backward'):
                # Decoder Grads
                decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, dec_param_scope.name)
                decoder_grads = list(zip(tf.gradients(self.loss,decoder_vars),decoder_vars))

                # Encoder Grads
                encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,enc_param_scope.name)
                grad_z = tf.gradients(self.recon,z_q)
                encoder_grads = [(tf.gradients(z_e,var,grad_z)[0]+beta*tf.gradients(self.commit,var)[0],var)
                                 for var in encoder_vars]
                # Embedding Grads
                embed_grads = list(zip(tf.gradients(self.vq,embeds),[embeds]))

                optimizer = tf.train.AdamOptimizer(lr)
                self.train_op= optimizer.apply_gradients(decoder_grads+ encoder_grads + embed_grads, global_step = global_step)
        else :
            # Another decoder pass that we can play with!
            self.latent = tf.placeholder(tf.int64,[None,3,3])
            _t = tf.gather(embeds, self.latent)
            for block in dec_spec:
                _t = block(_t)
            self.gen = _t

        all_save_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                param_scope.name) + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, dec_param_scope.name)

        save_vars = {('train/'+'/'.join(var.name.split('/')[1:])).split(':')[0] : var for var in all_save_vars }
        #for name,var in save_vars.items():
        #    print(name,var)

        self.saver = tf.train.Saver(var_list=save_vars)

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

    def save(self, sess, dir, step=None):
        if(step is not None):
            self.saver.save(sess, os.path.join(dir, 'model.ckpt'), global_step=step)
        else :
            self.saver.save(sess, os.path.join(dir, 'last.ckpt'))

    def load(self,sess,model):
        self.saver.restore(sess, model)


if __name__ == "__main__":
    with tf.variable_scope('params') as params:
        pass

    with open('wavenet_params.json', 'r') as f:
        wavenet_params = json.load(f)

    coord = tf.train.Coordinator()

    with tf.name_scope('create_inputs'):
        # Allow silence trimming to be skipped by specifying a threshold near
        # zero.
        silence_threshold = None

        #AUDIO_FILE_PATH = '/home/sriramso/data/VCTK-Corpus'
        AUDIO_FILE_PATH = '/home/andrewszot/VCTK-Corpus'

        gc_enabled = False
        reader = AudioReader(
            AUDIO_FILE_PATH,
            coord,
            sample_rate=wavenet_params['sample_rate'],
            gc_enabled=gc_enabled,
            receptive_field=WaveNetModel.calculate_receptive_field(wavenet_params["filter_width"],
                                                                   wavenet_params["dilations"],
                                                                   wavenet_params["scalar_input"],
                                                                   wavenet_params["initial_filter_width"]),
            sample_size=39939,
            silence_threshold=silence_threshold)

        audio_batch = reader.dequeue(1)
        if gc_enabled:
            gc_id_batch = reader.dequeue_gc(1)
        else:
            gc_id_batch = None

    global_step = tf.Variable(0, trainable=False)

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    reader.start_threads(sess)

    try:
        # 100K iterations
        MAX_STEPS = int(1e5) # We can move this to another file if we want
        log_dir = './logdir'
        learning_rate = 0.0001
        beta = 0.25


        net = VQVAE(learning_rate, global_step, beta, audio_batch, 380, 256, _audio_arch,
                sess, params, True)

        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)

        for step in xrange(int(MAX_STEPS)):
            start_time = time.time()
            _, overall_loss, recon_loss, vq_loss, commit_loss, z_q_mean, z_e_mean, input_mean, logits_mean, indices_mean, l0, l1, l2, d0, d1, d2, s0, s1, s2, s3, W_mean, b_mean = sess.run([net.train_op,
                net.loss, net.recon, net.vq, net.commit, net.z_q_mean,
                net.z_e_mean, net.input_mean, net.logits_mean,
                net.indices_mean, net.l0, net.l1, net.l2, net.d0, net.d1,
                net.d2, net.s0, net.s1, net.s2, net.s3, net.W_mean, net.b_mean])

            duration = time.time() - start_time
            print('')
            print('Step %d: overall = %.2f, recon = %.2f, vq = %.2f, commit = %.2f (%.3f sec)' %
                    (step, overall_loss, recon_loss, vq_loss, commit_loss, duration))
            print('z_q_mean: %.2f, z_e_mean: %.2f, input_mean: %.2f, logits_mean: %.2f, indices_mean: %.2f, ' % (z_q_mean, z_e_mean,
                input_mean, logits_mean, indices_mean))

            print('l0: %.2f, l1: %.2f, l2: %.2f, d0: %.2f, d1: %.2f, d2: %.2f, s0: %.2f, s1: %.2f, s2: %.2f, s3: %.2f, W: %.2f, b: %.2f' % (l0, l1, l2, d0,
                        d1, d2, s0, s1, s2, s3, W_mean, b_mean))

            if (step + 1) % 1000 == 0 or (step + 1) == MAX_STEPS:
                print('Saving!')
                net.save(sess, log_dir, step)
        #print(sess.run(net.x_scaled).shape)
        #print('Audio batch: ' + str(sess.run(audio_batch).shape))
        #print('z_q: ' + str(sess.run(net.z_q).shape))
        #print('Logits: ' + str(sess.run(net.logits).shape))
        #print('x_indices: ' + str(sess.run(net.x_indices).shape))
    except KeyboardInterrupt:
        # Introduce a line break after ^C is displayed so save message
        # is on its own line.
        print()
        # net.save(sess, log_dir, step)
    finally:
        coord.request_stop()
        coord.join(threads)


    #print(sess.run(net.train_op, feed_dict={x:np.random.random((10,32,32,1))}))

