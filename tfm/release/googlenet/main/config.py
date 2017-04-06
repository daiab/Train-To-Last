import logging
import os

net_name="google_net"

ps_hosts = ["10.29.150.102:2222"]
worker_hosts = ["10.29.150.102:2223", "10.29.150.102:2224", "10.29.150.102:2225", "10.29.150.102:2226",
                "10.29.150.103:2222", "10.29.150.103:2223", "10.29.150.103:2224","10.29.150.103:2225"]

# common config param
is_training = True
log_dir="log/"

data_dir="/home/mpiNode/data/"
tfrecords_filename_train = [data_dir + 'ms_train_data.tfrecords']
tfrecords_filename_test = [data_dir + 'test.tfrecords']

test_txt_file = "/home/daianbo/code/Filelist_LFW_5Pts.txt"
feature_txt_file = "/home/daianbo/data/feature.txt"


# hyper parameters
base_lr = 0.08
momentum = 0.9
power=0.6
weight_decay=0.0001
batch_size = 224


# model parameters
class_num = 10575


# train parameters
iter_num = 100000
decay_steps=iter_num * 1.1
print_loss_step = 40
queue_capacity = 10000
num_threads = 2
is_writer_summary=False
is_sync=True


def get_logger(file_name):
    logging.basicConfig(level=logging.DEBUG,
                              format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                              datefmt='%b %d %Y %H:%M:%S',
                              filename=log_dir + 'runtime/tensorflow.log',
                              filemode='w')
    return logging.getLogger(file_name)

def print_config():
    logger = get_logger(__name__)
    config_file_path = os.path.abspath(__file__)
    with open(config_file_path, 'r', encoding='utf-8') as config_file:
        for line in config_file:
            logger.info(" | " + line)



