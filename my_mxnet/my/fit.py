import mxnet as mx
import logging
import time
import os
import my_mxnet.my.config as cfg


def load_model(rank=0):
    if cfg.load_epoch is None:
        return (None, None, None)
    assert cfg.model_prefix is not None
    print("========  load model ========")
    model_prefix = cfg.model_prefix
    if rank > 0 and os.path.exists("%s-%d-symbol.json" % (model_prefix, rank)):
        model_prefix += "-%d" % (rank)
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, cfg.load_epoch)
    logging.info('Loaded model %s_%04d.params', model_prefix, cfg.load_epoch)
    return (sym, arg_params, aux_params)

def save_model(rank=0):
    if cfg.model_prefix is None:
        return None
    dst_dir = os.path.dirname(cfg.model_prefix)
    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)
    return mx.callback.do_checkpoint(cfg.model_prefix if rank == 0 else "%s-%d" % (
        cfg.model_prefix, rank))

def get_lr_scheduler(kv):
    if cfg.lr_factor is None or cfg.lr_factor >= 1:
        return (cfg.lr, None)
    epoch_size = cfg.num_examples / cfg.batch_size
    if 'dist' in cfg.kv_store:
        epoch_size /= kv.num_workers
    begin_epoch = cfg.load_epoch if cfg.load_epoch else 0
    lr = cfg.lr
    for s in cfg.lr_step_epochs:
        if begin_epoch >= s:
            lr *= cfg.lr_factor
    if lr != cfg.lr:
        logging.info('Adjust learning rate to %e for epoch %d' %(lr, begin_epoch))

    steps = [epoch_size * (x - begin_epoch) for x in cfg.lr_step_epochs if x - begin_epoch > 0]
    return (lr, mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=cfg.lr_factor))


def fit(network, data_loader, **kwargs):
    # kvstore
    print("build dist start")
    kv = mx.kvstore.create(cfg.kv_store)
    print("build dist over")
    # logging
    head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
                        #filename='/home/mpiNode/daiab/git/model/log/runtime/mxnet.log',
                        #filemode='w')
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


    # load model
    if 'arg_params' in kwargs and 'aux_params' in kwargs:
        arg_params = kwargs['arg_params']
        aux_params = kwargs['aux_params']
    else:
        sym, arg_params, aux_params = load_model(kv.rank)
        if sym is not None:
            assert sym.tojson() == network.tojson()

    # save model
    checkpoint = save_model(kv.rank)

    # devices for training
    devs = mx.cpu() if cfg.gpus is None or cfg.gpus is '' else \
        [mx.gpu(int(i)) for i in cfg.gpus.split(',')]

    # learning rate
    lr, lr_scheduler = get_lr_scheduler(kv)

    # create model
    model = mx.mod.Module(context=devs, symbol=network)

    optimizer_params = {
            'learning_rate': lr,
            'momentum': cfg.mom,
            'wd': cfg.wd,
            'lr_scheduler': lr_scheduler}

    monitor = mx.mon.Monitor(cfg.monitor, pattern=".*weight|learning_rate|softmax_label") if cfg.monitor > 0 else None

    if cfg.init_xavier:
        logging.info("init with xavier")
        initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2)
        # initializer   = mx.init.Xavier(factor_type="in", magnitude=2.34),
    else:
        # AlexNet will not converge using Xavier
        logging.info("init with normal")
        initializer = mx.init.Normal()


    # evaluation metrices
    eval_metrics = ['accuracy', 'cross-entropy']
    if cfg.top_k > 0:
        eval_metrics.append(mx.metric.create('top_k_accuracy', top_k=cfg.top_k))

    # callbacks that run after each batch
    batch_end_callbacks = [mx.callback.Speedometer(cfg.batch_size, cfg.disp_batches)]
    if 'batch_end_callback' in kwargs:
        cbs = kwargs['batch_end_callback']
        batch_end_callbacks += cbs if isinstance(cbs, list) else [cbs]

    # run
    model.fit(train,
        begin_epoch        = cfg.load_epoch if cfg.load_epoch else 0,
        num_epoch          = cfg.num_epochs,
        eval_data          = valid,
        eval_metric        = eval_metrics,
        kvstore            = kv,
        optimizer          = cfg.optimizer,
        optimizer_params   = optimizer_params,
        initializer        = initializer,
        arg_params         = arg_params,
        aux_params         = aux_params,
        batch_end_callback = batch_end_callbacks,
        epoch_end_callback = checkpoint,
        allow_missing      = True,
        monitor            = monitor)