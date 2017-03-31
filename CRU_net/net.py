from frame.base_net import BaseNet
import frame.config as cfg


class Net(BaseNet):
    def def_model(self):
        (self.feed('data')
         .conv2d(7, 7, 64, 2, 2, relu=True, bn= True, name='bn_conv_1')
         .max_pool(3, 3, 2, 2, name='pool_1'))

        # ----------1----------
        for i in range(0, 32, 1):
            (self.feed('pool_1')
             .conv2d(1, 1, 128, 1, 1, bias=False, relu=False, bn=False, name='1_branch_a_' + str(i))
             .conv2d(3, 3, 128, 1, 1, bias=False, relu=True, bn=False, name='1_branch_b_' + str(i))
             .conv2d(1, 1, 256, 1, 1, bias=False, relu=False, bn=True, name='1_branch_c_' + str(i))
             .register(BaseNet.SAVE))

        (self.register(BaseNet.READ)
         .add_n(relu=True, name="add_n_1")
         .conv2d(1, 1, 256, 1, 1, bias=False, relu=False, bn=False, name="conv_collect_1"))

        (self.feed("pool_1", "conv_collect_1")
         .add_n(relu=False, name="add_n_2")
         .bn(name="batch_norm_1")
         .relu(name="relu_1"))

        # ----------2----------
        for i in range(0, 32, 1):
            (self.feed('relu_1')
             .conv2d(1, 1, 256, 1, 1, bias=False, relu=False, bn=False, name='2_branch_a_' + str(i))
             .conv2d(3, 3, 256, 1, 1, bias=False, relu=True, bn=False, name='2_branch_b_' + str(i))
             .conv2d(1, 1, 512, 1, 1, bias=False, relu=False, bn=True, name='2_branch_c_' + str(i))
             .register(BaseNet.SAVE))

        (self.register(BaseNet.READ)
         .add_n(relu=True, name="add_n_3")
         .conv2d(1, 1, 512, 1, 1, bias=False, relu=False, bn=False, name="conv_collect_2"))

        (self.feed("relu_1", "conv_collect_2")
         .add_n(relu=False, name="add_n_4")
         .bn(name="batch_norm_2")
         .relu(name="relu_2"))

        # ----------3----------
        for i in range(0, 640, 1):
            (self.feed('relu_2')
             .conv2d(1, 1, 640, 1, 1, bias=False, relu=False, bn=False, name='3_branch_a_' + str(i))
             .conv2d(3, 3, 640, 1, 1, bias=False, relu=True, bn=False, name='3_branch_b_' + str(i))
             .conv2d(1, 1, 640, 1, 1, bias=False, relu=False, bn=True, name='3_branch_c_' + str(i))
             .conv2d(1, 1, 1024, 1, 1, bias=False, relu=False, bn=True, name='3_branch_d_' + str(i))
             .register(BaseNet.SAVE))

        (self.register(BaseNet.READ)
         .add_n(relu=True, name="add_n_5")
         .conv2d(1, 1, 256, 1, 1, bias=False, relu=False, bn=False, name="conv_collect_3"))

        (self.feed("relu_2", "conv_collect_3")
         .add_n(relu=False, name="add_n_6")
         .bn(name="batch_norm_3")
         .relu(name="relu_3"))

        # ----------4----------
        for i in range(0, 32, 1):
            (self.feed('relu_3')
             .conv2d(1, 1, 1024, 1, 1, bias=False, relu=False, bn=False, name='4_branch_a_' + str(i))
             .conv2d(3, 3, 1024, 1, 1, bias=False, relu=True, bn=False, name='4_branch_b_' + str(i))
             .conv2d(1, 1, 2048, 1, 1, bias=False, relu=False, bn=True, name='4_branch_c_' + str(i))
             .register(BaseNet.SAVE))

        (self.register(BaseNet.READ)
         .add_n(relu=True, name="add_n_7")
         .conv2d(1, 1, 256, 1, 1, bias=False, relu=False, bn=False, name="conv_collect_2"))

        (self.feed("relu_3", "conv_collect_4")
         .add_n(relu=False, name="add_n_8")
         .bn(name="batch_norm_4")
         .relu(name="relu_4")
         .avg_pool(7, 7, 1, 1, padding='VALID', name='pool5')
         .reshape(flatten=True, name="reshape_1")
         .wx_b([None, 1000], [1000], relu=False, bn=False, name="fc_1")
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
