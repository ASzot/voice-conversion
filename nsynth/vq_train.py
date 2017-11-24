from audio_reader import AudioReader
import tensorflow as tf
import json
from h512_bo16 import Config as ModelConfig

def calculate_receptive_field(filter_width, dilations, scalar_input,
                              initial_filter_width):
    receptive_field = (filter_width - 1) * sum(dilations) + 1
    if scalar_input:
        receptive_field += initial_filter_width - 1
    else:
        receptive_field += filter_width - 1
    return receptive_field

with open('wavenet_params.json', 'r') as f:
    wavenet_params = json.load(f)

coord = tf.train.Coordinator()

with tf.name_scope('create_inputs'):
    # Allow silence trimming to be skipped by specifying a threshold near
    # zero.
    silence_threshold = None

    #AUDIO_FILE_PATH = '/home/sriramso/data/VCTK-Corpus'
    #AUDIO_FILE_PATH = '/home/andrewszot/VCTK-Corpus'
    AUDIO_FILE_PATH = '/Users/andrewszot/Downloads/VCTK-Corpus'

    gc_enabled = False
    reader = AudioReader(
        AUDIO_FILE_PATH,
        coord,
        sample_rate=wavenet_params['sample_rate'],
        gc_enabled=gc_enabled,
        receptive_field=calculate_receptive_field(wavenet_params["filter_width"],
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
    MAX_STEPS = int(1e5) # We can move this to another file if we want
    log_dir = './logdir'
    learning_rate = 0.0001
    beta = 0.25

    model = ModelConfig()
    print('Preparing to build fowards')
    output = model.build(audio_batch, K=380, D=256, beta=0.25)

    print('Preparing to init variables')
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)
    print('Built okay!')

    # Actually train
    #for step in xrange(int(MAX_STEPS)):
    #    loss, _ = sess.run([output['loss'], output['train_op'])
    #    print('%i: %.2f' % loss)

except KeyboardInterrupt:
    # Introduce a line break after ^C is displayed so save message
    # is on its own line.
    print()
    # net.save(sess, log_dir, step)
finally:
    coord.request_stop()
    coord.join(threads)
