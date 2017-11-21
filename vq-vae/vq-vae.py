from six.moves import xrange
import better_exceptions
import tensorflow as tf
import numpy as np
from commons.ops import *
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

def _mnist_arch(d):
    with tf.variable_scope('enc') as enc_param_scope :
        enc_spec = [
            Conv2d('conv2d_1', 1, d // 4, data_format='NHWC'),
            lambda t,**kwargs : tf.nn.relu(t),
            Conv2d('conv2d_2',d // 4, d // 2, data_format='NHWC'),
            lambda t,**kwargs : tf.nn.relu(t),
            Conv2d('conv2d_3',d // 2, d, data_format='NHWC'),
            lambda t,**kwargs : tf.nn.relu(t),
        ]
    with tf.variable_scope('dec') as dec_param_scope :
        dec_spec = [
            TransposedConv2d('tconv2d_1', d, d//2, data_format='NHWC'),
            lambda t,**kwargs : tf.nn.relu(t),
            TransposedConv2d('tconv2d_2', d//2, d//4, data_format='NHWC'),
            lambda t,**kwargs : tf.nn.relu(t),
            TransposedConv2d('tconv2d_3', d//4, 1, data_format='NHWC'),
            lambda t,**kwargs : tf.nn.sigmoid(t),
        ]
    return enc_spec, enc_param_scope, dec_spec, dec_param_scope


class VQVAE():
    """
    lr: learning rate.
    x: input tensor.
    K: Number of embeddings.
    D: Dimension of embedding.
    """
    def __init__(self, lr, global_step, beta,
                 x,K,D,
                 arch_fn,
                 is_1d,
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
            #x_scaled = tf.expand_dims(x_scaled, 2)
            #self.x_scaled = x_scaled

            _t = x_scaled
            for block in enc_spec :
                _t = block(_t)
            z_e = _t
            self.tmp_z_e = z_e

            # Middle Area (Compression or Discretize)
            # TODO: Gross.. use brodcast instead!

            if is_1d:
                tile_arr = [1, 1, K, 1]
                reshape_arr = [1, 1, K, D]
            else:
                tile_arr = [1, 1, 1, K, 1]
                reshape_arr = [1, 1, K, D]
            _t = tf.tile(tf.expand_dims(z_e, -2),tile_arr) #[batch,latent_h,latent_w,K,D]
            _e = tf.reshape(embeds, reshape_arr)
            _t = tf.norm(_t - _e, axis = -1)
            k = tf.argmin(_t, axis = -1) # -> [latent_h,latent_w]
            self.k = k
            z_q = tf.gather(embeds, k)

            self.z_e = z_e # -> [batch,latent_h,latent_w,D]
            self.k = k
            self.z_q = z_q # -> [batch,latent_h,latent_w,D]

            #print(sess.run(self.z_e).shape)
            #print(sess.run(self.z_q).shape)
            #raise ValueError()

            # End early
            return

            # Decoder Pass
            _t = z_q
            for block in dec_spec:
                _t = block(_t)
            self.p_x_z = _t

            # Losses
            self.recon = tf.reduce_mean((self.p_x_z - x) ** 2, axis=[0,1,2,3])
            self.vq = tf.reduce_mean(
                tf.norm(tf.stop_gradient(self.z_e) - z_q, axis=-1) ** 2,
                axis=[0,1,2])
            self.commit = tf.reduce_mean(
                tf.norm(self.z_e - tf.stop_gradient(z_q), axis=-1) ** 2,
                axis=[0,1,2])
            self.loss = self.recon + self.vq + beta * self.commit

            # NLL
            # TODO: is it correct impl?
            # it seems tf.reduce_prod(tf.shape(self.z_q)[1:2]) should be multipled
            # in front of log(1/K) if we assume uniform prior on z.
            self.nll = -1.*(tf.reduce_mean(tf.log(self.p_x_z),axis=[1,2,3]) + tf.log(1/tf.cast(K,tf.float32)))/tf.log(2.)

        if( is_training ):
            with tf.variable_scope('backward'):
                # Decoder Grads
                decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,dec_param_scope.name)
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

        save_vars = {('train/'+'/'.join(var.name.split('/')[1:])).split(':')[0] : var for var in
                     tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,param_scope.name) }
        #for name,var in save_vars.items():
        #    print(name,var)

        self.saver = tf.train.Saver(var_list=save_vars,max_to_keep = 3)

    def save(self, sess, dir, step=None):
        if(step is not None):
            self.saver.save(sess, dir + '/model.ckpt', global_step=step)
        else :
            self.saver.save(sess, dir + '/last.ckpt')

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

        gc_enabled = False
        reader = AudioReader(
            '/hdd/cs599/VCTK-Corpus',
            coord,
            sample_rate=wavenet_params['sample_rate'],
            gc_enabled=gc_enabled,
            receptive_field=WaveNetModel.calculate_receptive_field(wavenet_params["filter_width"],
                                                                   wavenet_params["dilations"],
                                                                   wavenet_params["scalar_input"],
                                                                   wavenet_params["initial_filter_width"]),
            sample_size=100000,
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

        is_2d = True
        net = VQVAE(0.1, global_step, 0.25, audio_batch, 380, 256, _audio_arch,
                is_2d, sess, params, True)

        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)

        #print(sess.run(net.x_scaled).shape)
        print(sess.run(audio_batch).shape)
        print(sess.run(net.tmp_z_e).shape)
        print(sess.run(net.z_e).shape)
        print(sess.run(net.z_q).shape)
        print(sess.run(net.k).shape)
    except KeyboardInterrupt:
        # Introduce a line break after ^C is displayed so save message
        # is on its own line.
        print()
    finally:
        coord.request_stop()
        coord.join(threads)




    #print(sess.run(net.train_op, feed_dict={x:np.random.random((10,32,32,1))}))

