import mxnet as mx
import my_mxnet.my.config as cfg


def get_rec_iter(kv=None):
    if kv:
        (rank, nworker) = (kv.rank, kv.num_workers)
    else:
        (rank, nworker) = (0, 1)
    train = mx.io.ImageRecordIter(
        path_imgrec         = cfg.data_train,
        label_width         = 1,
        mean_r              = cfg.rgb_mean[0],
        mean_g              = cfg.rgb_mean[1],
        mean_b              = cfg.rgb_mean[2],
        data_name           = 'data',
        label_name          = 'softmax_label',
        data_shape          = cfg.image_shape,
        batch_size          = cfg.batch_size,
        rand_crop           = cfg.random_crop,
        # max_random_scale    = cfg.max_random_scale,
        # min_random_scale    = cfg.min_random_scale,
        # max_aspect_ratio    = cfg.max_random_aspect_ratio,
        # random_h            = cfg.max_random_h,
        # random_s            = cfg.max_random_s,
        # random_l            = cfg.max_random_l,
        # max_rotate_angle    = cfg.max_random_rotate_angle,
        # max_shear_ratio     = cfg.max_random_shear_ratio,
        rand_mirror         = cfg.random_mirror,
        preprocess_threads  = cfg.data_nthreads,
        shuffle             = False,
        num_parts           = nworker,
        part_index          = rank)
    if cfg.data_valid is None:
        return (train, None)
    valid = mx.io.ImageRecordIter(
        path_imgrec         = cfg.data_valid,
        label_width         = 1,
        mean_r              = cfg.rgb_mean[0],
        mean_g              = cfg.rgb_mean[1],
        mean_b              = cfg.rgb_mean[2],
        data_name           = 'data',
        label_name          = 'softmax_label',
        batch_size          = cfg.batch_size,
        data_shape          = cfg.image_shape,
        preprocess_threads  = cfg.data_nthreads,
        rand_crop           = False,
        rand_mirror         = False,
        num_parts           = nworker,
        part_index          = rank)
    return (train, valid)
