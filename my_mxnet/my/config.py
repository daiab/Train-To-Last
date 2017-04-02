# load the model on an epoch using the model-load-prefix
load_epoch=None
# model prefix
model_prefix="log/train/model-"
# train record data path
data_train="/home/mpiNode/data/img.rec"
# the validation data
data_valid="/home/mpiNode/data/img.rec"
# initial learning rate
lr=0.05
# the ratio to reduce lr on each step
lr_factor=0.5
# the batch size
batch_size=896
# key-value store type
kv_store="local" #"dist"
# the epochs to reduce the lr, e.g. 30,60
lr_step_epochs=(10000, 20000, 40000)
# 1 means test reading speed without training
test_io=False
# show progress for every n batches
disp_batches=40
# list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu
gpus="0,1,2,3"
# momentum  for sgd
mom=0.9
# weight decay for sgd
wd=0.0001
# log network parameters every N iters if larger than 0
monitor=100
#the neural network to use
network="googlenet"
# report the top-k accuracy. 0 means no report.
top_k=4
# the optimizer type default='sgd'
optimizer='sgd'
# max num of epochs
num_epochs=40
# number workers
num_workers = 4


# read_data
# padding the input image
pad_size=0
# the number of classes
num_classes=10575
# the number of training examples
num_examples=390000
# the image shape feed into the network
image_shape=(3, 224, 224)
# a tuple of size 3 for the mean rgb
rgb_mean=[255, 255, 255]
# if or not randomly crop the image
random_crop=1
max_random_l=0
random_mirror=1
data_nthreads=4

