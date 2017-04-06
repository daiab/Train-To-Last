from tfm.snapshot import BaseNet
from tfm.snapshot import frame as cfg


class Net(BaseNet):
    def def_model(self):
        (self.feed("images_1", "image_2")
         .concat(name="concat_image")
         .conv2d(5, 5, 48, 2, 2, name="conv_1")
         .max_pool(3, 3, 2, 2, name="pool_1")
         .conv2d(1, 1, 96, 1, 1, name="conv_2")
         .conv2d(3, 3, 96, 1, 1, name="conv_3")
         .conv2d(1, 1, 96, 1, 1, name="conv_4")
         .conv2d(3, 3, 96, 1, 1, name="conv_5")
         .max_pool(3, 3, 2, 2, name="pool_2")
         .conv2d(1, 1, 192, 1, 1, name="conv_6")
         .conv2d(3, 3, 192, 1, 1, name="conv_7")
         .conv2d(1, 1, 192, 1, 1, name="conv_8")
         .conv2d(3, 3, 192, 1, 1, name="conv_9")
         .conv2d(1, 1, 192, 1, 1, name="conv_10")
         .conv2d(3, 3, 192, 1, 1, name="conv_11")
         .max_pool(3, 3, 2, 2, name="pool_3"))

        (self.feed("pool_3")
         .reshape(flatten=True, name="reshape")
         .wx_b([None, 4096], [4096], name="fc_1")
         .dropout(0.5, name="dropout_1")
         .wx_b([4096, 2048], [2048], name="fc_2")
         .dropout(0.5, name="dropout_2")
         .wx_b([2048, cfg.class_num], [cfg.class_num], relu=False, bn=False, name="fc_3")
         .split(name="split_fc")
         .fetch("batch_1", "batch_2"))

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