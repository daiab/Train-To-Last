# python im2rec.py /home/daiab/Pictures image_fold  --exts=.PNG
# /home/daiab/Pictures: the image_fold path
# image_fold: the fold name of containing image
# /home/daiab/Pictures path must contain a .lst file which describe the image path and label

# read test after write
import mxnet as mx
import matplotlib.pyplot as plt
import numpy as np

batch_size = 10

data_iter = mx.io.ImageRecordIter(
    #path_imgrec="/home/daiab/Pictures/test.rec", # the target record file
    path_imgrec="/home/mpiNode/data/img.rec", # the target record file
    data_shape=(3, 256, 256), # output data shape. An 227x227 region will be cropped from the original image.
    batch_size=batch_size # number of samples per batch
    # resize=256 # resize the shorter edge to 256 before cropping
    # ... you can add more augumentation options here. use help(mx.io.ImageRecordIter) to see all possible choices
    )

show_image = False

data_iter.reset()
batch = data_iter.next()
if show_image:
    plt.ion()
    print(len(batch.data))
    for batch_idx in range(batch_size):
        data = batch.data[0][batch_idx]
        plt.imshow(data.asnumpy().astype(np.uint8).transpose((1,2,0)))
        plt.show()
        plt.pause(0.01)
else:
    print("one sample shape = ", batch.data[0][0].asnumpy().shape)
    print(batch.label[0].asnumpy())

