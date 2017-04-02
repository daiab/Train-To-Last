import mxnet as mx

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


import time
batch_size = 1024
shape = [batch_size, 1, 28, 28]
mnist.reset()
tic = time.time()
acc = train(lenet(), shape, lambda:mnist.get(batch_size), [mx.gpu(),], [1,])
assert acc > 0.8, "Low training accuracy."
print('time for train lenent on cpu %f sec' % (time.time() - tic))


# @@@ AUTOTEST_OUTPUT_IGNORED_CELL
for ndev in (2, 4):
    mnist.reset()
    tic = time.time()
    acc = train(lenet(), shape, lambda:mnist.get(batch_size),
          [mx.gpu(i) for i in range(ndev)], [1]*ndev)
    assert acc > 0.9, "Low training accuracy."
    print('time for train lenent on %d GPU %f sec' % (
            ndev, time.time() - tic))


