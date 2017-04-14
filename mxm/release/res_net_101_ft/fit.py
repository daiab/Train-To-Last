import logging
import os
import time

import mxnet as mx

import mxm.release.res_net_101_ft.config as cfg


def load_model(rank=0):
    if cfg.load_epoch is None:
        return (None, None, None)
    assert cfg.model_prefix is not None
    print("========  load model ========")
    model_prefix = cfg.model_prefix
    if rank > 0 and os.path.exists("%s-%d-symbol.json" % (model_prefix, rank)):
        model_prefix += "-%d" % (rank)
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, cfg.load_epoch)
    logging.info('Loaded model ========= %s_%04d.params', model_prefix, cfg.load_epoch)
    return (sym, arg_params, aux_params)

def save_model(rank=0):
    if cfg.model_prefix is None:
        return None
    dst_dir = os.path.dirname(cfg.model_prefix)
    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)
    return mx.callback.do_checkpoint(cfg.model_prefix if rank == 0 else "%s-%d" % (
        cfg.model_prefix, rank))


#TODO: rescale_grad?
sgd_opt = mx.optimizer.SGD(learning_rate=cfg.lr, momentum=cfg.mom, wd=cfg.wd, rescale_grad=1/cfg.batch_size)
def lr_callback(param):
    global_nbatch = param.epoch * int(cfg.num_examples / cfg.batch_size) + param.nbatch
    sgd_opt.lr = (1 - global_nbatch / cfg.decay_nbatch) ** cfg.pow * (cfg.lr - cfg.end_lr) + cfg.end_lr
    if param.nbatch % cfg.disp_batches == 0:
        logging.info('Epoch[%d] Batch [%d]	learning rate:%f' % (param.epoch, param.nbatch, sgd_opt.lr))

def fit(network, data_loader, **kwargs):
    # kvstore
    print("build dist start")
    kv = mx.kvstore.create(cfg.kv_store)
    print("build dist over")
    # logging
    head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    logging.info('start with arguments %s', cfg)

    # data iterators
    (train, valid) = data_loader(kv)
    if cfg.test_io:
        tic = time.time()
        for i, batch in enumerate(train):
            for j in batch.data:
                j.wait_to_read()
            if (i+1) % cfg.disp_batches == 0:
                logging.info('Batch [%d]\tSpeed: %.2f samples/sec' % (
                    i, cfg.disp_batches*cfg.batch_size/(time.time()-tic)))
                tic = time.time()
        return


    if 'arg_params' in kwargs and 'aux_params' in kwargs:
        arg_params = kwargs['arg_params']
        aux_params = kwargs['aux_params']
    else:
        sym, arg_params, aux_params = load_model(kv.rank)
        if sym is not None:
            assert sym.tojson() == network.tojson()

    checkpoint = save_model(kv.rank)

    # devices for training
    devs = mx.cpu() if cfg.gpus is None or len(cfg.gpus) == 0 \
        else [mx.gpu(int(i)) for i in cfg.gpus]

    model = mx.mod.Module(context=devs, symbol=network,
                          data_names=["data_ms", "data_cnfd"], label_names=["label_ms", "label_cnfd"])

    if cfg.init_xavier:
        logging.info("init with xavier")
        initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2)
        # initializer   = mx.init.Xavier(factor_type="in", magnitude=2.34),
    else:
        # AlexNet will not converge using Xavier
        logging.info("init with normal")
        initializer = mx.init.Normal()


    # evaluation metrices
    eval_metrics = ['accuracy', 'ce']
    # if cfg.top_k > 0:
    #     eval_metrics.append(mx.metric.create('top_k_accuracy', top_k=cfg.top_k))

    batch_end_callbacks = [mx.callback.Speedometer(cfg.batch_size, cfg.disp_batches), lr_callback]
    if 'batch_end_callback' in kwargs:
        cbs = kwargs['batch_end_callback']
        batch_end_callbacks += cbs if isinstance(cbs, list) else [cbs]

    # run
    model.fit(train,
        begin_epoch        = cfg.begin_epoch,
        num_epoch          = cfg.num_epochs,
        eval_data          = valid,
        eval_metric        = eval_metrics,
        kvstore            = kv,
        optimizer          = sgd_opt,
        initializer        = initializer,
        arg_params         = arg_params,
        aux_params         = aux_params,
        batch_end_callback = batch_end_callbacks,
        epoch_end_callback = checkpoint,
        allow_missing      = True)
