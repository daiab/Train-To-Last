import mxm.release.res_net_101_ft.read_data as data
import mxm.release.res_net_101_ft.res_net as net
import mxnet as mx

import mxm.release.res_net_101_ft.config as cfg
from mxm.release.res_net_101_ft import fit
from mxm.release.res_net_101_ft import predict

if cfg.is_train:
    print("=======train model========")
    sym = net.Symble(input={"data_ms": mx.symbol.Variable(name="data_ms")
                            , "data_cnfd": mx.symbol.Variable(name="data_cnfd")},
                     input_shape=(int(cfg.batch_size / cfg.num_workers), 3, 224, 224))
    total_loss = mx.symbol.Group([sym.layer['softmax_ms'], sym.layer['softmax_cnfd']])
    #https://github.com/hariag/mxnet-multi-task-example/blob/master/multi-task.ipynb
    fit.fit(total_loss, data.get_rec_iter)
else:
    print("=======test model========")
    sym = net.Symble(input={"data": mx.symbol.Variable(name="data")},
                     input_shape=(int(cfg.batch_size / cfg.num_workers), 3, 224, 224))
    predict.predict(sym.layer['softmax_m'], data.get_rec_iter_test)


# export DMLC_ROLE=worker; export DMLC_NUM_SERVER=4 ;python my_mxnet/my/train.py
# dist train:
# http://mxnet.io/how_to/multi_devices.html