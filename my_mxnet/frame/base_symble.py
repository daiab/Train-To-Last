import mxnet as mx

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
        print("---------layer info-----------")
        print(name)
        if isinstance(layer_value, mx.symbol):
            print(layer_value.infer_shape(data=self.input_shape))
        # print(layer_value)
        layer_output = op(self, layer_value, *args, **kwargs)
        self.layer[name] = layer_output
        self.feed(name)
        return self

    return layer_decorated


class BaseSymble(object):
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

    def batch_norm(self, inputs, decay=0.99, epsilon=1e-7, name="batch_norm"):
        return mx.symbol.BatchNorm(data=inputs, eps=epsilon, momentum=decay, fix_gamma=False)

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
            pad_left = int(pad / 2)
            pad_right = pad - pad_left
        else:
            pad_left = pad_right = 0
        print("pad_left = %d, pad_right = %d" %(pad_left, pad_right))
        conv = mx.symbol.Convolution(data=input, num_filter=out_num, kernel=(kernel_h, kernel_w),
                                     stride=(stride_h, stride_w), pad=(pad_left, pad_right),
                                     no_bias=not bias, name=name)
        if bn:
            conv = self.batch_norm(conv)
        if relu:
            conv = mx.symbol.Activation(data=conv, act_type='relu', name='activation')
        return conv

    def lrn(self,
            input,
            radius,
            alpha,
            beta,
            name):
        """
        data (Symbol) – Input data to the ConvolutionOp.
        alpha (float, optional, default=0.0001) – value of the alpha variance scaling parameter in the normalization formula
        beta (float, optional, default=0.75) – value of the beta power parameter in the normalization formula
        knorm (float, optional, default=2) – value of the k parameter in normalization formula
        nsize (int (non-negative), required) – normalization window width in elements.
        name (string, optional.) – Name of the resulting symbol.
        """
        return mx.symbol.LRN(data=input, nsize=radius, alpha=alpha, beta=beta)

    @layer
    def bn(self, inputs, decay=0.99, epsilon=1e-7, name="batch_norm"):
        return mx.symbol.BatchNorm(data=inputs, eps=epsilon, momentum=decay, fix_gamma=False)

    @layer
    def relu(self, input, name="relu"):
        return mx.symbol.Activation(data=input, act_type='relu', name='activation')

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
            pad_right = pad - pad_left
        else:
            pad_left = pad_right = 0
        return mx.symbol.Pooling(data=input, pool_type='max', kernel=(kernel_h, kernel_w),
                                 stride=(stride_h, stride_w), pad=(pad_left, pad_right), name=name)

    @layer
    def avg_pool(self,
                 input,
                 kernel_h,
                 kernel_w,
                 stride_h,
                 stride_w,
                 padding=(0, 0),
                 name='SAME'):
        if padding == 'SAME':
            inf_shape = input.infer_shape(data=self.input_shape)[1][0][0]
            tmp = inf_shape % stride_w
            pad = kernel_w - tmp
            pad_left = int(pad / 2)
            pad_right = pad - pad_left
        else:
            pad_left = pad_right = 0
        return mx.symbol.Pooling(data=input, pool_type='avg', kernel=(kernel_h, kernel_w),
                                 stride=(stride_h, stride_w), pad=(pad_left, pad_right), name=name)

    @layer
    def concat(self, input, axis=0, name="concat"):
        return mx.symbol.Concat(*input, dim=axis, name='concat-' + name)

    @layer
    def fc(self, input, out_num, name="fc"):
        flatten = mx.sym.Flatten(data=input, name="flatten-" + name)
        return mx.sym.FullyConnected(data=flatten, num_hidden=out_num, name="full-connect-" + name)

    @layer
    def softmax(self, inputs, name="softmax"):
        softmax = mx.symbol.SoftmaxOutput(data=inputs, name=name)
        return softmax
