import logging
import os
import mxnet as mx
import mxm.release.res_net_50.config as cfg

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


def read_file_name(txt_file_path):
    # /home/daiab/test/test.csv
    f = open(txt_file_path, 'r')
    file_dir = []
    for line in f:
        filename = line.split(' ')[0]
        file_dir.append(filename)
    f.close()
    return file_dir


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

    sym_feature = all_layers['Flatten-fc_output']
    mod_feature = mx.mod.Module(symbol=sym_feature, context=mx.gpu())
    mod_feature.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))])
    mod_feature.set_params(arg_params, aux_params)
    fo=open(cfg.ext_feat_name, 'w')
    idx = 0
    file_name_list = read_file_name(cfg.test_lst_name)
    file_list_len = len(file_name_list)
    data_iter = data_loader()
    data_iter.reset()
    while True:
        data_test = data_iter.next()
        if data_test is None:
            break
        if idx > file_list_len:
            logging.info("error idx out of range, last idx file_name = %s, idx = %d" % (file_name_list[idx-1], idx))
            break
        mod_feature.forward(data_test)
        extract_feature = mod_feature.get_outputs()[0].asnumpy()
        feat_str = list(extract_feature[0].astype(str))
        fo.write("%s %s\n" %(file_name_list[idx], " ".join(feat_str)))
        idx += 1
        if idx % 1000 == 0:
            logging.info(extract_feature.shape)
        logging.info("index: %d" % idx)
    fo.close()
