import mxnet as mx
from functools import reduce

def layer(op):
    def layer_decorated(self, *args, **kwargs):
        # name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        name = kwargs['name']
        if len(self.top) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.top) == 1:
            layer_input_name = self.top[0]
            layer_value = self.layer[layer_input_name]
        else:
            layer_value = [self.layer[key] for key in self.top]
        # print(layer_value)
        layer_output = op(self, layer_value, *args, **kwargs)
        self.layer[name] = layer_output
        print("---------layer info-----------")
        print(name)
        if isinstance(layer_output, mx.symbol.Symbol):
            print(layer_output.infer_shape(data=self.input_shape))
        self.feed(name)
        return self

    return layer_decorated


class BaseSymble(object):
    SAVE = 0
    READ = 1

    def __init__(self, input=None, input_shape=None):
        """
        input: like {'input_layer_name': input_layer_value}
        layer: like {'layer_name': layer_value}
        top: like ['layer_name_1', 'layer_name_2', ...]
        """
        self.layer = input
        self.input_shape = input_shape
        self.top = []
        self.register = []
        self.def_model()

    def def_model(self):
        '''Construct the network. '''
        raise NotImplementedError('Must be implemented by the subclass.')

    def feed(self, *args):
        assert len(args) != 0, ("feed args cound not be None")
        self.top = []
        for fed_name in args:
            self.top.append(fed_name)
        return self

    @layer
    def conv2d(self,
               input,
               kernel_h,
               kernel_w,
               out_num,
               stride_h,
               stride_w,
               padding='SAME',
               bn=True,
               relu=True,
               bias=True,
               name='conv2d'):
        if padding == 'SAME':
            inf_shape = input.infer_shape(data=self.input_shape)[1][0][0]
            tmp = inf_shape % stride_w
            pad = kernel_w - tmp
            pad_right = pad_left = int(pad / 2)
            # pad_right = pad - pad_left
        else:
            pad_left = pad_right = 0
        print("pad_left = %d, pad_right = %d" %(pad_left, pad_right))
        conv = mx.symbol.Convolution(data=input, num_filter=out_num, kernel=(kernel_h, kernel_w),
                                     stride=(stride_h, stride_w), pad=(pad_left, pad_right),
                                     no_bias=not bias, name="Convolution-%s" %name)
        if bn:
            print("conv use batch norm")
            conv = mx.symbol.BatchNorm(data=conv, fix_gamma=False, eps=2e-5, momentum=0.99, name="BatchNorm-%s" %name)
        if relu:
            conv = mx.symbol.Activation(data=conv, act_type='relu', name="Activation-%s" %name)
        return conv

    def fetch(self, *args):
        assert len(args) != 0, ("out args cound not be None")
        assert len(self.top) == len(args), ("out args length should equals top length")
        for i in range(len(args)):
            self.layer[args[i]] = self.top[i]
        self.top = []

    def register(self, save_or_read):
        """save or read   save: 0; read: 1"""
        if save_or_read == self.SAVE:
            self.register.extend(self.top)
            return self
        elif save_or_read == self.READ:
            self.top = self.register
            self.register = []
            return self

    @layer
    def bn(self, inputs, decay=0.99, epsilon=2e-5, name="batch_norm"):
        return mx.symbol.BatchNorm(data=inputs, eps=epsilon, momentum=decay, fix_gamma=False, name="BatchNorm-%s" %name)

    @layer
    def relu(self, input, name="relu"):
        return mx.symbol.Activation(data=input, act_type='relu', name="Activation-%s" %name)

    @layer
    def max_pool(self,
                 input,
                 kernel_h,
                 kernel_w,
                 stride_h,
                 stride_w,
                 padding='SAME',
                 name="max_pool"):
        if padding == 'SAME':
            inf_shape = input.infer_shape(data=self.input_shape)[1][0][0]
            tmp = inf_shape % stride_w
            pad = kernel_w - tmp
            pad_left = int(pad / 2)
            pad_right = pad_left = int(pad / 2)
            # pad_right = pad - pad_left
        else:
            pad_left = pad_right = 0
        return mx.symbol.Pooling(data=input, pool_type='max', kernel=(kernel_h, kernel_w),
                                 stride=(stride_h, stride_w), pad=(pad_left, pad_right), name="MaxPooling-%s" %name)

    @layer
    def avg_pool(self,
                 input,
                 kernel_h,
                 kernel_w,
                 stride_h,
                 stride_w,
                 padding='VALID',
                 name='avg_pool'):
        if padding == 'SAME':
            inf_shape = input.infer_shape(data=self.input_shape)[1][0][0]
            tmp = inf_shape % stride_w
            pad = kernel_w - tmp
            pad_left = int(pad / 2)
            pad_right = pad_left = int(pad / 2)
            # pad_right = pad - pad_left
        else:
            pad_left = pad_right = 0
        return mx.symbol.Pooling(data=input, pool_type='avg', kernel=(kernel_h, kernel_w),
                                 stride=(stride_h, stride_w), pad=(pad_left, pad_right), name="AvgPooling-%s" %name)

    @layer
    def concat(self, input, axis=0, name="concat"):
        return mx.symbol.Concat(*input, dim=axis, name="Concat-%s" %name)

    @layer
    def split(self, input, axis=0, num_output=2, name="split"):
        return mx.symbol.split(data=input, axis=axis, num_output=num_output, name=name)

    def any_split(self, input, cut_point, ndim,  axis=0, name="anysplit"):
        """
        x = [[ 1., 2., 3., 4.],[ 5., 6., 7., 8.], [ 9., 10., 11., 12.]]
            slice_axis(x, axis=0, begin=1, end=3) = [[ 5., 6., 7., 8.],[ 9., 10., 11., 12.]]
        """
        slice_down = mx.symbol.slice_axis(data=input, axis=axis, begin=0, end=cut_point, name="slicedonw-%s" % name)
        slice_up = mx.symbol.slice_axis(data=input, axis=axis, begin=cut_point, end=ndim, name="sliceup-%s" % name)
        return [slice_down, slice_up]

    @layer
    def fc(self, input, out_num, flatten=False, relu=True, bn=False, name="fc"):
        if flatten: input = mx.sym.Flatten(data=input, name="Flatten-%s" %name)
        out = mx.sym.FullyConnected(data=input, num_hidden=out_num, name="FullyConnected-%s" %name)
        if bn: out = mx.symbol.BatchNorm(data=out, fix_gamma=False, eps=2e-5, momentum=0.99, name="BatchNorm-%s" %name)
        if relu: out = mx.symbol.Activation(data=out, act_type='relu', name="Activation-%s" %name)
        return out

    @layer
    def softmax(self, inputs, name="softmax"):
        softmax = mx.symbol.SoftmaxOutput(data=inputs, name="softmax")
        return softmax

    @layer
    def add_n(self, input, relu=False, name="add_n"):
        if isinstance(input, list):
            out = mx.symbol.add_n(*input)
            # out = reduce(lambda a, b: a + b, input)
        else:
            out = input
        if relu:
            return mx.symbol.Activation(data=out, act_type='relu', name="Activation-%s" %name)
        else:
            return out
