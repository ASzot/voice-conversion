import tensorflow as tf

NUM_THREADS=4

def get_image_batch(pattern, batch_size, train=True) :
    if (train) :
        random_flip = lambda x : tf.image.random_flip_left_right(x)
        crop = lambda x : tf.random_crop(x,[crop_size,crop_size,3])
        queue = lambda : tf.train.string_input_producer(tf.train.match_filenames_once(pattern),
                                                         num_epochs=None, shuffle=True)
        batch = lambda f,x: tf.train.shuffle_batch([f,x],
                                                    batch_size=batch_size,
                                                    num_threads=NUM_THREADS,
                                                    capacity=batch_size*5,
                                                    min_after_dequeue=batch_size*3)
    else :
        random_flip = lambda x : tf.identity(x)
        crop = lambda x : tf.image.resize_image_with_crop_or_pad(image,crop_size,crop_size)
        queue = lambda : tf.train.string_input_producer(tf.train.match_filenames_once(pattern),
                                                         num_epochs=1, shuffle=False)
        batch = lambda f,x: tf.train.batch([f,x],
                                            batch_size=batch_size,
                                            num_threads=NUM_THREADS,
                                            allow_smaller_final_batch=False)

    def _preprocess(image) :
        #image = random_flip(image)
        #image = crop(image)
        image = tf.transpose(image,[2,0,1]) #change to CHW format
        image = (tf.cast(image,tf.float32) - 128.0)/128.0 #push in to [-1 to 1] area.
        return image

    with tf.device('/cpu:0'):
        filename_queue = queue()

        image_reader = tf.WholeFileReader()
        filename, image_file = image_reader.read(filename_queue)
        image = tf.image.decode_jpeg(image_file, 3)

        image.set_shape((258, 258, 3))
        # Ignore the 3rd dimension, that was only used as padding to save to
        # jpeg.
        #image = tf.slice(image, [0, 0, 0], [258, 258, 2])
        #print('Image shape %s' % str(image.get_shape()))
        preprocessed = _preprocess(image)

        filenames, images = batch(filename,preprocessed)

    return filenames, images
