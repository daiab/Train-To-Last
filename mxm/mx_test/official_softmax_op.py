import os
import mxnet as mx
import numpy as np

class Softmax(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        x = in_data[0].asnumpy()
        y = np.exp(x - x.max(axis=1).reshape((x.shape[0], 1)))
        y /= y.sum(axis=1).reshape((x.shape[0], 1))
        self.assign(out_data[0], req[0], mx.nd.array(y))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        l = in_data[1].asnumpy().ravel().astype(np.int)
        y = out_data[0].asnumpy()
        y[np.arange(l.shape[0]), l] -= 1.0
        self.assign(in_grad[0], req[0], mx.nd.array(y))


@mx.operator.register("softmax")
class SoftmaxProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(SoftmaxProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = (in_shape[0][0],)
        output_shape = in_shape[0]
        return [data_shape, label_shape], [output_shape], []

    def infer_type(self, in_type):
        dtype = in_type[0]
        return [dtype, dtype], [dtype], []

    def create_operator(self, ctx, shapes, dtypes):
        return Softmax()


mlp = mx.symbol.Custom(data=fc3, name='softmax', op_type='softmax')
a = mx.sym.Variable('a')
a.bind()

mod = mx.mod.Module(symbol=net,
                    context=mx.cpu(),
                    data_names=['data'],
                    label_names=['softmax_label'])
mx.model.load_checkpoint()
mod.fit()
mx.io