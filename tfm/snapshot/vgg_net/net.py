from tfm.snapshot import BaseNet
from tfm.snapshot import frame as cfg


class Net(BaseNet):
    def def_model(self):
        (self.feed('data')
         .conv(3, 3, 64, 1, 1, name='conv1_1')
         .conv(3, 3, 64, 1, 1, name='conv1_2')
         .max_pool(2, 2, 2, 2, name='pool1')
         .conv(3, 3, 128, 1, 1, name='conv2_1')
         .conv(3, 3, 128, 1, 1, name='conv2_2')
         .max_pool(2, 2, 2, 2, name='pool2')
         .conv(3, 3, 256, 1, 1, name='conv3_1')
         .conv(3, 3, 256, 1, 1, name='conv3_2')
         .conv(3, 3, 256, 1, 1, name='conv3_3')
         .max_pool(2, 2, 2, 2, name='pool3')
         .conv(3, 3, 512, 1, 1, name='conv4_1')
         .conv(3, 3, 512, 1, 1, name='conv4_2')
         .conv(3, 3, 512, 1, 1, name='conv4_3')
         .max_pool(2, 2, 2, 2, name='pool4')
         .conv(3, 3, 512, 1, 1, name='conv5_1')
         .conv(3, 3, 512, 1, 1, name='conv5_2')
         .conv(3, 3, 512, 1, 1, name='conv5_3')
         .max_pool(2, 2, 2, 2, name='pool5')
         .reshape(flatten=True, name="reshape")
         .wx_b([None, 4096], [4096], name="fc_1")
         .wx_b([4096, 4096], [4096], name="fc_2")
         .wx_b([4096, 1000], [1000], relu=False, bn=False, name="fc_3")
         .split(name="split_fc")
         .fetch("batch_1", "batch_2")
         )

        (self.feed("batch_1", "labels_1")
         .cross_entropy(name="cross_entropy_1")
         .loss_plus_norm(weight_decay=cfg.weight_decay, name="loss_plus_norm_1"))

        (self.feed("batch_2", "labels_2")
         .cross_entropy(name="cross_entropy_2")
         .loss_plus_norm(weight_decay=cfg.weight_decay, name="loss_plus_norm_2"))

        (self.feed("batch_1", "labels_1")
         .accuracy(name="accuracy_1"))

        (self.feed("batch_2", "labels_2")
         .accuracy(name="accuracy_2"))
