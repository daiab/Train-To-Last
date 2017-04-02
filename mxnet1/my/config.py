# load the model on an epoch using the model-load-prefix
load_epoch=0
# model prefix
model_prefix="log/train/model-"
# train record data path
data_train=""
# initial learning rate
lr=0
# the ratio to reduce lr on each step
lr_factor=0.5
# the batch size
batch_size=0
# key-value store type
kv_store="dist"
# the epochs to reduce the lr, e.g. 30,60
lr_step_epochs=(30, 60, 90)
# 1 means test reading speed without training
test_io=0
# show progress for every n batches
disp_batches=40
# list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu
gpus=0,1,2,3
# momentum  for sgd
mom=0
# weight decay for sgd
wd=0
# log network parameters every N iters if larger than 0
monitor=0
#the neural network to use
network=0
# report the top-k accuracy. 0 means no report.
top_k=0
# the optimizer type default='sgd'
optimizer=0
# max num of epochs
num_epochs=0
# number workers
num_workers = 1


# read_data
# padding the input image
pad_size=0
# the number of classes
num_classes=0
# the number of training examples
num_examples=0
# the image shape feed into the network
image_shape=(3,224,224)
# a tuple of size 3 for the mean rgb
rgb_mean=123.68,116.779,103.939
# if or not randomly crop the image
random_crop=1
# max ratio to scale
max_random_scale=1
# min ratio to scale, should >= img_size/input_shape. otherwise use --pad-size
min_random_scale=0
# max change of aspect ratio, whose range is [0, 1]
max_random_aspect_ratio=0
# max change of hue, whose range is [0, 180]
max_random_h=0
# max change of saturation, whose range is [0, 255]
max_random_s=0
# max change of intensity, whose range is [0, 255]
max_random_l=0
# max angle to rotate, whose range is [0, 360]
max_random_rotate_angle=0
# max ratio to shear, whose range is [0, 1]
max_random_shear_ratio=0
# if or not randomly flip horizontally
random_mirror=0
data_nthreads=0
# the validation data
data_val=0
