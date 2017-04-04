is_train=True
# load the model on an epoch using the model-load-prefix
load_epoch=None
# model prefix
model_prefix="log/train/model"
# train record data path
data_train="/home/mpiNode/data/img.rec"
# test record data path
data_test="/home/mpiNode/data/img.rec"
# the validation data
data_valid=None #"/home/mpiNode/data/img.rec"
# initial learning rate
lr=0.05
pow=0.5
end_lr=0.0001
decay_nbatch=60000
# the ratio to reduce lr on each step
# lr_factor=0.5
# the batch size
batch_size=800
# key-value store type
kv_store="local" #"dist"
# 1 means test reading speed without training
test_io=False
# show progress for every n batches
disp_batches=40
# list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu
gpus=[0, 1, 2, 3]
# number workers
num_workers = len(gpus)
# momentum  for sgd
mom=0.9
# weight decay for sgd
wd=0.0001
# log network parameters every N iters if larger than 0
monitor=200
#the neural network to use
init_xavier=True
# report the top-k accuracy. 0 means no report.
# top_k=0
# max num of epochs
num_epochs=100



# read_data
# padding the input image
pad_size=0
# the number of classes
num_classes=10575
# the number of training examples
num_examples=469216
# the image shape feed into the network
image_shape=(3, 224, 224)
# a tuple of size 3 for the mean rgb
rgb_mean=[255, 255, 255]
# if or not randomly crop the image
random_crop=1
max_random_l=0
random_mirror=1
data_nthreads=8

