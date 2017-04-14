import mxnet as mx

import mxm.release.res_net_50.config as cfg


def get_rec_iter(kv=None):
    if kv:
        (rank, nworker) = (kv.rank, kv.num_workers)
    else:
        (rank, nworker) = (0, 1)
    train = mx.io.ImageRecordIter(
        path_imgrec         = cfg.data_train,
        label_width         = 1,
        scale=1.0 / 255,
        data_name           = 'data',
        label_name          = 'softmax_label',
        data_shape          = cfg.image_shape,
        batch_size          = cfg.batch_size,
        rand_crop           = cfg.random_crop,
        rand_mirror         = cfg.random_mirror,
        preprocess_threads  = cfg.data_nthreads,
        shuffle             = False,
        num_parts           = nworker,
        part_index          = rank,
        prefetch_buffer     = 10)
    if cfg.data_valid is None:
        return (train, None)
    valid = mx.io.ImageRecordIter(
        path_imgrec         = cfg.data_valid,
        label_width         = 1,
        scale=1.0 / 255,
        data_name           = 'data',
        label_name          = 'softmax_label',
        batch_size          = cfg.batch_size,
        data_shape          = cfg.image_shape,
        preprocess_threads  = cfg.data_nthreads,
        rand_crop           = False,
        rand_mirror         = False,
        num_parts           = nworker,
        part_index          = rank,
        prefetch_buffer     = 10)
    return (train, valid)

def get_rec_iter_test(batch_size=1):
    test = mx.io.ImageRecordIter(
        path_imgrec=cfg.data_test,
        label_width=1,
        scale=1.0/255,
        data_name='data',
        # label_name='softmax_label',
        batch_size=batch_size,
        data_shape=cfg.image_shape,
        preprocess_threads=cfg.data_nthreads,
        rand_crop=False,
        rand_mirror=False)
    return test
