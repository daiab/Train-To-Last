# import my_mxnet.my.googlenet as net
import my_mxnet.my.googlenet as net
import my_mxnet.my.read_data as data
from my_mxnet.my import fit
import mxnet as mx

sym = net.Symble({"data": mx.symbol.Variable(name="data")})
fit.fit(sym, data.get_rec_iter)


# export DMLC_ROLE=worker; export DMLC_NUM_SERVER=4 ;python my_mxnet/my/train.py