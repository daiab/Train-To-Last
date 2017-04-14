import numpy as np
from tfm.release.googlenet.frame import *
from tensorflow.python.training import moving_averages

from tfm.release.googlenet.main import config as cfg

NORM_KEY = 'NORM_KEY'

logger = cfg.get_logger('net_info')


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
        print(layer_value)
        # print(layer_value)
        layer_output = op(self, layer_value, *args, **kwargs)
        self.layer[name] = layer_output
        self.feed(name)
        return self

    return layer_decorated


def summary(sum_op):
    def summary_decorated(self, s_name, value, layer_name):
        if cfg.is_writer_summary:
            logger.info("summary name == %s the correspond layer name == %s", s_name, layer_name)
            return sum_op(self, s_name, value)
        else:
            pass

    return summary_decorated


class BaseNet(object):
    SAVE=0
    READ=1

    def __init__(self, input=None):
        """
        input: like {'input_layer_name': input_layer_value}
        layer: like {'layer_name': layer_value}
        top: like ['layer_name_1', 'layer_name_2', ...]
        """
        self.input = input
        self.layer = input
        self.top = []
        self.register = []
        self.def_model()

    def get_losses(self, key="losses", scope=None):
        losses = tf.get_collection(key=key, scope=scope)
        assert len(losses) == 1
        print(losses[0])
        return losses[0]

    def feed(self, *args):
        assert len(args) != 0, ("feed args cound not be None")
        self.top = []
        for fed_name in args:
            self.top.append(fed_name)
        return self

    def register(self, save_or_read):
        """save or read   save: 0; read: 1"""
        if save_or_read == self.SAVE:
            self.register.extend(self.top)
            return self
        elif save_or_read == self.READ:
            self.top = self.register
            self.register = []
            return self

    def fetch(self, *args):
        assert len(args) != 0, ("out args cound not be None")
        assert len(self.top) == len(args), ("out args length should equals top length")
        for i in range(len(args)):
            self.layer[args[i]] = self.top[i]
        self.top = []

    def def_model(self):
        '''Construct the network. '''
        raise NotImplementedError('Must be implemented by the subclass.')

    def batch_norm(self, inputs, decay=0.995, epsilon=1e-7, name="batch_norm"):
        shape = inputs.get_shape()[-1]
        axis = list(range(len(inputs.get_shape()) - 1))
        gamma = tf.get_variable(name=name + "-gamma", shape=[shape], dtype=tf.float32,
                                initializer=tf.ones_initializer())
        beta = tf.get_variable(name=name + "-beta", shape=[shape], dtype=tf.float32,
                               initializer=tf.zeros_initializer())
        global_mean = tf.get_variable(name=name + "-global_mean", shape=[shape], dtype=tf.float32,
                                      initializer=tf.zeros_initializer(), trainable=False)
        global_var = tf.get_variable(name=name + "-global_var", shape=[shape], dtype=tf.float32,
                                     initializer=tf.ones_initializer(), trainable=False)
        if cfg.is_training:
            batch_mean, batch_var = tf.nn.moments(inputs, axis)
            batch_mean.set_shape((shape,))
            batch_var.set_shape((shape,))
            train_mean = moving_averages.assign_moving_average(global_mean, batch_mean, decay)
            train_var = moving_averages.assign_moving_average(global_var, batch_var, decay)

            self.histogram_summary(name + "-summary_global_mean", global_mean, name)
            self.histogram_summary(name + "-summary_global_var", global_var, name)

            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, gamma, epsilon)
        else:
            return tf.nn.batch_normalization(inputs, global_mean, global_var, beta, gamma, epsilon)

    def add_run_metadata_summary(self, summary_writer, sess, global_step, merged_summary):
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, i = sess.run([merged_summary, global_step],
                              options=run_options,
                              run_metadata=run_metadata)
        summary_writer.add_run_metadata(run_metadata, 'step%d' % i)
        summary_writer.add_summary(summary, global_step)

    @layer
    def conv2d(self,
               input,
               kernel_h,
               kernel_w,
               out_num,
               stride_h,
               stride_w,
               padding='SAME',
               mean=0.0,
               stddev=None,
               bn=True,
               relu=True,
               bias=True,
               const_value=0.0,
               name='conv2d',
               add_norm=True):
        stddev = stddev or np.sqrt(2 / np.prod(input.get_shape().as_list()[1:]))
        in_num = input.get_shape()[3]
        w = tf.get_variable(name=name + "-conv2d_w", shape=[kernel_h, kernel_w, in_num, out_num], dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(mean=mean, stddev=stddev, dtype=tf.float32))
        bias_var = tf.get_variable(name=name + "-conv2d_b", shape=[out_num], dtype=tf.float32,
                               initializer=tf.constant_initializer(value=const_value, dtype=tf.float32))
        if add_norm: tf.add_to_collection(NORM_KEY, value=w)
        self.histogram_summary(name + "-summary_w", w, name)
        self.histogram_summary(name + "-summary_bias", bias, name)
        tmp = tf.nn.conv2d(input=input, filter=w, strides=[1, stride_h, stride_w, 1], padding=padding, name=name)
        if bias:
            tmp += bias_var
        if bn:
            tmp = self.batch_norm(tmp, name=name + '-batch_norm')
        if relu:
            tmp = tf.nn.relu(tmp)
        return tmp

    @layer
    def bn(self, inputs, decay=0.999, epsilon=1e-7, name="batch_norm"):
        shape = inputs.get_shape()[-1]
        axis = list(range(len(inputs.get_shape()) - 1))
        gamma = tf.get_variable(name=name + "-gamma", shape=[shape], dtype=tf.float32,
                                initializer=tf.ones_initializer())
        beta = tf.get_variable(name=name + "-beta", shape=[shape], dtype=tf.float32,
                               initializer=tf.zeros_initializer())
        global_mean = tf.get_variable(name=name + "-global_mean", shape=[shape], dtype=tf.float32,
                                      initializer=tf.zeros_initializer(), trainable=False)
        global_var = tf.get_variable(name=name + "-global_var", shape=[shape], dtype=tf.float32,
                                     initializer=tf.ones_initializer(), trainable=False)
        if cfg.is_training:
            batch_mean, batch_var = tf.nn.moments(inputs, axis)
            batch_mean.set_shape((shape,))
            batch_var.set_shape((shape,))
            train_mean = moving_averages.assign_moving_average(global_mean, batch_mean, decay)
            train_var = moving_averages.assign_moving_average(global_var, batch_var, decay)

            self.histogram_summary(name + "-summary_global_mean", global_mean, name)
            self.histogram_summary(name + "-summary_global_var", global_var, name)

            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, gamma, epsilon)
        else:
            return tf.nn.batch_normalization(inputs, global_mean, global_var, beta, gamma, epsilon)

    @layer
    def relu(self, input, name="relu"):
        return tf.nn.relu(input, name=name)

    @layer
    def max_pool(self,
                 input,
                 kernel_h,
                 kernel_w,
                 stride_h,
                 stride_w,
                 padding='SAME',
                 name="max_pool"):
        return tf.nn.max_pool(value=input, ksize=[1, kernel_h, kernel_w, 1], strides=[1, stride_h, stride_w, 1],
                              padding=padding, name=name)

    @layer
    def avg_pool(self,
                 input,
                 kenerl_h,
                 kenerl_w,
                 stride_h,
                 stride_w,
                 padding='VALID',
                 name="avg_pool"):
        return tf.nn.avg_pool(value=input, ksize=[1, kenerl_h, kenerl_w, 1], strides=[1, stride_h, stride_w, 1],
                              padding=padding, name=name)

    @layer
    def lrn(self,
            input,
            radius,
            alpha,
            beta,
            name,
            bias=1.0):
        return tf.nn.lrn(input,
                         depth_radius=radius,
                         alpha=alpha,
                         beta=beta,
                         bias=bias,
                         name=name)

    @layer
    def wx_b(self,
             x,
             w_shape,
             b_shape,
             add_norm=True,
             relu=True,
             bn=True,
             name="wx_b"):
        # stddev = np.sqrt(2 / np.prod(x.get_shape().as_list()[1:]))
        if w_shape[0] is None:
            w_shape[0] = x.get_shape().as_list()[-1]
        w = tf.get_variable(name=name + "-wx_b_w", shape=w_shape, dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
                            # initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
        b = tf.get_variable(name=name + "-wx_b_b", shape=b_shape, dtype=tf.float32,
                            initializer=tf.constant_initializer())
        if add_norm: tf.add_to_collection(NORM_KEY, value=w)
        self.histogram_summary(name + "-summary_w", w, name)
        self.histogram_summary(name + "-summary_b", b, name)
        inner_pro = tf.nn.xw_plus_b(x, w, b)
        if bn:
            inner_pro = self.batch_norm(inner_pro, name=name + "-batch_norm")
        if relu:
            return tf.nn.relu(inner_pro)

        return inner_pro

    @layer
    def reshape(self, x, shape=None, flatten=False, name="reshape"):
        """if flatten is False , shape is needed"""
        if flatten:
            length = np.prod(x.get_shape().as_list()[1:])
            shape = [-1, length]
        return tf.reshape(x, shape=shape, name=name)

    @layer
    def dropout(self, x, keep_prob=0.5, name="dropout"):
        return tf.nn.dropout(x, keep_prob=keep_prob, name=name)

    @layer
    def cross_entropy(self, logit_and_label, name="cross_entropy"):
        ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logit_and_label[0], labels=logit_and_label[1], name=name))
        self.scalar_summary(name + "-cross_entropy", ce, name)
        return ce

    @layer
    def loss_plus_norm(self, loss, weight_decay=0.0004, scope=None, norm_type="l2", name="losses"):
        """
        norm_type: could be l1, l2 or others
        """
        weight = tf.get_collection(key=NORM_KEY, scope=scope)
        print("---------norm weight-----------")
        for w in weight:
            print(w.name)
        assert norm_type in ["l2", "l1"]
        if norm_type == "l2":
            norm = tf.add_n([tf.nn.l2_loss(i) for i in weight])
        else:
            norm = tf.multiply(weight_decay, tf.reduce_sum(tf.abs(weight)))
        total_loss = loss + weight_decay * norm
        # self._losses = total_loss
        tf.add_to_collection(name="losses", value=total_loss)
        self.scalar_summary(name + "-summary_losses", total_loss, name)
        return total_loss

    @layer
    def accuracy(self, fc_out_and_label, name="accuracy"):
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(fc_out_and_label[0], 1), fc_out_and_label[1]), tf.float32),
                             name=name)
        self.scalar_summary(name + "-summary_accuracy", acc, name)
        return acc

    @layer
    def split(self, input, name="split"):
        """split cross batch size into two batches"""
        batch_size = input.get_shape()[0]
        assert batch_size % 2 == 0, ("split into two batch size erros")
        return tf.split(input, [batch_size / 2, batch_size / 2], 0, name=name)

    @layer
    def concat(self, input, axis=0, name="concat"):
        return tf.concat(input, axis=axis, name=name)

    @layer
    def add_n(self, input, name="add_n", relu=False):
        if relu:
            return tf.nn.relu(tf.add_n(inputs=input, name=name))
        else:
            return tf.add_n(inputs=input, name=name)

    @summary
    def scalar_summary(self, s_name, value):
        return tf.summary.scalar(s_name, value)

    @summary
    def histogram_summary(self, s_name, value):
        return tf.summary.histogram(s_name, value)

