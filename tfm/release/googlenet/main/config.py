import logging
import os

net_name="google_net"

ps_hosts = ["10.29.150.102:2222"]
worker_hosts = ["10.29.150.102:2223", "10.29.150.102:2224", "10.29.150.102:2225", "10.29.150.102:2226",
                "10.29.150.103:2222", "10.29.150.103:2223", "10.29.150.103:2224","10.29.150.103:2225"]

# common config param
is_training = True
log_dir="tfm/log/"

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
class_num = 41857
# the number of training examples
num_examples=3095536
# max num of epochs
num_epochs=8
# train parameters
iter_num = num_epochs * int(num_examples / batch_size)
decay_steps=iter_num
print_loss_step = 40
queue_capacity = batch_size * 10
num_threads = 8
is_writer_summary=False
is_sync=True


def get_logger(file_name):
    logging.basicConfig(level=logging.DEBUG,
                              format='%(asctime)s %(filename)s[line:%(lineno)d] %(message)s',
                              datefmt='%b %d %Y %H:%M:%S',
                              filename='%sruntime/%s.log' %(log_dir, net_name),
                              filemode='w')
    return logging.getLogger(file_name)

def print_config():
    logger = get_logger(__name__)
    config_file_path = os.path.abspath(__file__)
    with open(config_file_path, 'r', encoding='utf-8') as config_file:
        for line in config_file:
            logger.info(" | " + line)



