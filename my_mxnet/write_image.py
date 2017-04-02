import mxnet as mx

data_iter = mx.io.ImageRecordIter(
    path_imgrec="./data/caltech_train.rec", # the target record file
    data_shape=(3, 227, 227), # output data shape. An 227x227 region will be cropped from the original image.
    batch_size=4, # number of samples per batch
    resize=256 # resize the shorter edge to 256 before cropping
    # ... you can add more augumentation options here. use help(mx.io.ImageRecordIter) to see all possible choices
    )
data_iter.reset()
batch = data_iter.next()
data = batch.data[0]
for i in range(4):
    plt.subplot(1,4,i+1)
    plt.imshow(data[i].asnumpy().astype(np.uint8).transpose((1,2,0)))
plt.show()


# pack_img
data = np.ones((3,3,1), dtype=np.uint8)
label = 1.0
header = mx.recordio.IRHeader(flag=0, label=label, id=0, id2=0)
s = mx.recordio.pack_img(header, data, quality=100, img_fmt='.jpg')
print(repr(s))

# unpack_img
print(*mx.recordio.unpack_img(s))