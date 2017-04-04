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
    logging.info('Loaded model ========= %s_%04d.params', model_prefix, cfg.load_epoch)
    return (sym, arg_params, aux_params)


def predict(network, data_loader):
    print("start to predict ...... ")
    # logging
    head = '%(asctime)-15s Node[0] %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    logging.info('start with arguments %s', cfg)

    sym, arg_params, aux_params = load_model()
    assert sym is not None and sym.tojson() == network.tojson()

    all_layers = sym.get_internals()
    print("===== symbol internals ====")
    print(all_layers.list_outputs()[-10:-1])

    sym_feature = all_layers['full-connect-fc']
    mod_feature = mx.mod.Module(symbol=sym_feature, context=mx.gpu())
    mod_feature.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))])
    mod_feature.set_params(arg_params, aux_params)

    data_loader.reset()
    data_test = data_loader.next()
    while True:
        if data_test is None:
            break
        mod_feature.forward(data_test)
        extract_feature = mod_feature.get_outputs()[0].asnumpy()
        logging.info(extract_feature.shape)
        data_test = data_loader.next()

