import logging


# common config param
is_training = True
log_dir="/home/daiab/Documents/log/"
data_dir="/home/daiab/data/"
save_dir="/home/daiab/save_model/"

tfrecords_filename_train = [data_dir + 'train.tfrecords']
tfrecords_filename_test = [data_dir + 'test.tfrecords']

test_txt_file = "/home/daianbo/code/Filelist_LFW_5Pts.txt"
feature_txt_file = "/home/daianbo/data/feature.txt"
save_model_path = save_dir + "tf_model.ckpt"


# hyper parameters
base_lr = 0.03
momentum = 0.9
power=0.5
weight_decay=0.0001
batch_size = 448


# model parameters
class_num = 10575


# train parameters
iter_num = 80000
decay_steps=iter_num
print_loss_step = 40
save_model_step = 5000
queue_capacity = 200
num_threads = 2
is_writer_summary=False
summary_dir=log_dir + "summary"
is_sync = True


def get_logger(file_name):
    logging.basicConfig(level=logging.DEBUG,
                              format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                              datefmt='%b %d %Y %H:%M:%S',
                              filename=log_dir + 'tensorflow.log',
                              filemode='w')
    return logging.getLogger(file_name)



