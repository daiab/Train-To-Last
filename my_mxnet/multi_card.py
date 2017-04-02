import mxnet as mx


def train(network, data_shape, data, devs, devs_power):
    # partition the batch into each device
    batch_size = float(data_shape[0])
    workloads = [int(round(batch_size/sum(devs_power)*p)) for p in devs_power]
    print('workload partition: ', zip(devs, workloads))
    # create an executor for each device
    exs = [network.simple_bind(ctx=d, data=tuple([p]+data_shape[1:])) for d, p in zip(devs, workloads)]
    args = [dict(zip(network.list_arguments(), ex.arg_arrays)) for ex in exs]
    # initialize weight on dev 0
    for name in args[0]:
        arr = args[0][name]
        if 'weight' in name:
            arr[:] = mx.random.uniform(-0.1, 0.1, arr.shape)
        if 'bias' in name:
            arr[:] = 0
    # run 50 iterations
    learning_rate = 0.1
    acc = 0
    for i in range(50):
        # broadcast weight from dev 0 to all devices
        for j in range(1, len(devs)):
            for name, src, dst in zip(network.list_arguments(), exs[0].arg_arrays, exs[j].arg_arrays):
                if 'weight' in name or 'bias' in name:
                    src.copyto(dst)
        # get data
        x, y = data()
        for j in range(len(devs)):
            # partition and assign data
            idx = range(sum(workloads[:j]), sum(workloads[:j+1]))
            args[j]['data'][:] = x[idx,:].reshape(args[j]['data'].shape)
            args[j]['out_label'][:] = y[idx].reshape(args[j]['out_label'].shape)
            # forward and backward
            exs[j].forward(is_train=True)
            exs[j].backward()
            # sum over gradient on dev 0
            if j > 0:
                for name, src, dst in zip(network.list_arguments(), exs[j].grad_arrays, exs[0].grad_arrays):
                    if 'weight' in name or 'bias' in name:
                        dst += src.as_in_context(dst.context)
        # update weight on dev 0
        for weight, grad in zip(exs[0].arg_arrays, exs[0].grad_arrays):
            weight[:] -= learning_rate * (grad / batch_size)
        # monitor
        if i % 10 == 0:
            pred = np.concatenate([mx.nd.argmax_channel(ex.outputs[0]).asnumpy() for ex in exs])
            acc = (pred == y).sum() / batch_size
            print('iteration %d, accuracy %f' % (i, acc))
    return acc


def lenet():
    data = mx.sym.Variable('data')
    # first conv
    conv1 = mx.sym.Convolution(data=data, kernel=(5,5), num_filter=20)
    tanh1 = mx.sym.Activation(data=conv1, act_type="tanh")
    pool1 = mx.sym.Pooling(data=tanh1, pool_type="max",
                           kernel=(2,2), stride=(2,2))
    # second conv
    conv2 = mx.sym.Convolution(data=pool1, kernel=(5,5), num_filter=50)
    tanh2 = mx.sym.Activation(data=conv2, act_type="tanh")
    pool2 = mx.sym.Pooling(data=tanh2, pool_type="max",
                           kernel=(2,2), stride=(2,2))
    # first fullc
    flatten = mx.sym.Flatten(data=pool2)
    fc1 = mx.sym.FullyConnected(data=flatten, num_hidden=500)
    tanh3 = mx.sym.Activation(data=fc1, act_type="tanh")
    # second fullc
    fc2 = mx.sym.FullyConnected(data=tanh3, num_hidden=10)
    # loss
    lenet = mx.sym.SoftmaxOutput(data=fc2, name='out')
    return lenet
mx.viz.plot_network(lenet(), shape={'data':(128,1,28,28)})

from sklearn.datasets import fetch_mldata
import numpy as np
import matplotlib.pyplot as plt


class MNIST:
    def __init__(self):
        mnist = fetch_mldata('MNIST original')
        p = np.random.permutation(mnist.data.shape[0])
        self.X = mnist.data[p]
        self.Y = mnist.target[p]
        self.pos = 0

    def get(self, batch_size):
        p = self.pos
        self.pos += batch_size
        return self.X[p:p + batch_size, :], self.Y[p:p + batch_size]

    def reset(self):
        self.pos = 0

    def plot(self):
        for i in range(10):
            plt.subplot(1, 10, i + 1)
            plt.imshow(self.X[i].reshape((28, 28)), cmap='Greys_r')
            plt.axis('off')
        plt.show()


mnist = MNIST()
mnist.plot()



#We first train lenet on a single GPU

# @@@ AUTOTEST_OUTPUT_IGNORED_CELL
import time
batch_size = 1024
shape = [batch_size, 1, 28, 28]
mnist.reset()
tic = time.time()
acc = train(lenet(), shape, lambda:mnist.get(batch_size), [mx.gpu(),], [1,])
assert acc > 0.8, "Low training accuracy."
print('time for train lenent on cpu %f sec' % (time.time() - tic))


# Then we try multiple GPUs. The following codes needs 4 GPUs.

# @@@ AUTOTEST_OUTPUT_IGNORED_CELL
for ndev in (2, 4):
    mnist.reset()
    tic = time.time()
    acc = train(lenet(), shape, lambda:mnist.get(batch_size),
          [mx.gpu(i) for i in range(ndev)], [1]*ndev)
    assert acc > 0.9, "Low training accuracy."
    print('time for train lenent on %d GPU %f sec' % (
            ndev, time.time() - tic))


