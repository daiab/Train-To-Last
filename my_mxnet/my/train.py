# import my_mxnet.my.googlenet as net
import my_mxnet.my.vgg as net
import my_mxnet.my.read_data as data
from my_mxnet.my import fit

sym = net.get_symbol()
fit.fit(sym, data.get_rec_iter)
