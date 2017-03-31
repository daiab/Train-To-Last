from release.googlenet.frame import *
from release.googlenet.main import config as cfg


def distort_color(image, scope=None):
    """
    see@ /home/daiab/ocode/models/inception/inception/image_processing.py
    """
    with tf.name_scope(values=[image], name=scope, default_name='distort_color'):
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)

        # The random_* ops do not necessarily clamp.
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image


# TODO: use multi threads and batch join
def read_and_decode(tfrecord_file_dir, crop_size=224):
    # tfrecord_file_dir = cfg.tfrecords_filename_train if cfg.is_training else cfg.tfrecords_filename_test
    data_queue = tf.train.string_input_producer(tfrecord_file_dir, capacity=cfg.queue_capacity, name="string_input_producer")

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(data_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string)
        })

    image_raw = tf.decode_raw(features['image_raw'], tf.uint8)
    label = features['label']

    image = tf.reshape(image_raw, [256, 256, 3])
    image.set_shape([256, 256, 3])
    image_scalar = tf.cast(image, tf.float32) / 255.0

    if cfg.is_training:
        # random crop image to 80*80*3
        # convert to NCHW
        # tf.transpose(image_scalar, [2, 0, 1])
        random_crop = tf.random_crop(image_scalar, [crop_size, crop_size, 3])
        image_random_flip = tf.image.random_flip_left_right(random_crop)
        distort_image = distort_color(image_random_flip)

        image_batch, label_batch = tf.train.batch([distort_image, label],
                                                     batch_size=cfg.batch_size,
                                                     capacity=cfg.queue_capacity,
                                                     num_threads=cfg.num_threads)
                                                     #enqueue_many=True)
        return image_batch, label_batch
    else:
        tf.image.crop_to_bounding_box(image_scalar, 16, 16, crop_size, crop_size)
        return tf.expand_dims(image_scalar, 0), tf.expand_dims(label, 0)