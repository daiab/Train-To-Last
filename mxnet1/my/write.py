# python im2rec.py /home/daiab/Pictures image_fold  --exts=.png --train-ratio=1 --test-ratio=0 --shuffle=1 --quality=100
# /home/daiab/Pictures: the image_fold path
# image_fold: the fold name of containing image
# /home/daiab/Pictures path must contain a .lst file which describe the image path and label

# read test after write
import mxnet as mx
import matplotlib.pyplot as plt
import numpy as np

data_iter = mx.io.ImageRecordIter(
    path_imgrec="./test.rec", # the target record file
    data_shape=(3, 256, 256), # output data shape. An 227x227 region will be cropped from the original image.
    batch_size=4 # number of samples per batch
    # resize=256 # resize the shorter edge to 256 before cropping
    # ... you can add more augumentation options here. use help(mx.io.ImageRecordIter) to see all possible choices
    )

data_iter.reset()
batch = data_iter.next()
data = batch.data[0]
print(batch.label[0].asnumpy())
for i in range(4):
    plt.subplot(1, 4, i+1)
    plt.imshow(data[i].asnumpy().astype(np.uint8).transpose((1,2,0)))
plt.show()
