import mxm.release.res_net_50.config as cfg
from mxm.release.base_symble import BaseSymble


class Symble(BaseSymble):
    def def_model(self):
        (self.feed('data')
         .conv2d(7, 7, 64, 2, 2, relu=True, bn=True, name='bn_conv1')
         .max_pool(3, 3, 2, 2, name='pool1')
         # ----------div----------
         .conv2d(1, 1, 256, 1, 1,  bias=False, relu=False, bn=True, name='bn2a_branch1'))

        (self.feed('pool1')
         .conv2d(1, 1, 64, 1, 1,  bias=False, relu=True, bn=True, name='bn2a_branch2a')
         .conv2d(3, 3, 64, 1, 1,  bias=False, relu=True, bn=True, name='bn2a_branch2b')
         .conv2d(1, 1, 256, 1, 1,  bias=False, relu=False, bn=True, name='bn2a_branch2c'))

        (self.feed('bn2a_branch1',
                   'bn2a_branch2c')
         .add_n(relu=True, name='res2a_relu')
         # ----------div----------
         .conv2d(1, 1, 64, 1, 1,  bias=False, relu=True, bn=True, name='bn2b_branch2a')
         .conv2d(3, 3, 64, 1, 1,  bias=False, relu=True, bn=True, name='bn2b_branch2b')
         .conv2d(1, 1, 256, 1, 1,  bias=False, relu=False, bn=True, name='bn2b_branch2c'))

        (self.feed('res2a_relu',
                   'bn2b_branch2c')
         .add_n(relu=True, name='res2b_relu')
         # ----------div----------
         .conv2d(1, 1, 64, 1, 1,  bias=False, relu=True, bn=True, name='bn2c_branch2a')
         .conv2d(3, 3, 64, 1, 1,  bias=False, relu=True, bn=True, name='bn2c_branch2b')
         .conv2d(1, 1, 256, 1, 1,  bias=False, relu=False, bn=True, name='bn2c_branch2c'))

        (self.feed('res2b_relu',
                   'bn2c_branch2c')
         .add_n(relu=True, name='res2c_relu')
         # ----------div----------
         .conv2d(1, 1, 512, 2, 2,  bias=False, relu=False, bn=True, name='bn3a_branch1'))

        (self.feed('res2c_relu')
         .conv2d(1, 1, 128, 2, 2,  bias=False, relu=True, bn=True, name='bn3a_branch2a')
         .conv2d(3, 3, 128, 1, 1,  bias=False, relu=True, bn=True, name='bn3a_branch2b')
         .conv2d(1, 1, 512, 1, 1,  bias=False, relu=False, bn=True, name='bn3a_branch2c'))

        (self.feed('bn3a_branch1',
                   'bn3a_branch2c')
         .add_n(relu=True, name='res3a_relu')
         # ----------div----------
         .conv2d(1, 1, 128, 1, 1,  bias=False, relu=True, bn=True, name='bn3b_branch2a')
         .conv2d(3, 3, 128, 1, 1,  bias=False, relu=True, bn=True, name='bn3b_branch2b')
         .conv2d(1, 1, 512, 1, 1,  bias=False, relu=False, bn=True, name='bn3b_branch2c'))

        (self.feed('res3a_relu',
                   'bn3b_branch2c')
         .add_n(relu=True, name='res3b_relu')
         # ----------div----------
         .conv2d(1, 1, 128, 1, 1,  bias=False, relu=True, bn=True, name='bn3c_branch2a')
         .conv2d(3, 3, 128, 1, 1,  bias=False, relu=True, bn=True, name='bn3c_branch2b')
         .conv2d(1, 1, 512, 1, 1,  bias=False, relu=False, bn=True, name='bn3c_branch2c'))

        (self.feed('res3b_relu',
                   'bn3c_branch2c')
         .add_n(relu=True, name='res3c_relu')
         # ----------div----------
         .conv2d(1, 1, 128, 1, 1,  bias=False, relu=True, bn=True, name='bn3d_branch2a')
         .conv2d(3, 3, 128, 1, 1,  bias=False, relu=True, bn=True, name='bn3d_branch2b')
         .conv2d(1, 1, 512, 1, 1,  bias=False, relu=False, bn=True, name='bn3d_branch2c'))

        (self.feed('res3c_relu',
                   'bn3d_branch2c')
         .add_n(relu=True, name='res3d_relu')
         # ----------div----------
         .conv2d(1, 1, 1024, 2, 2,  bias=False, relu=False, bn=True, name='bn4a_branch1'))

        (self.feed('res3d_relu')
         .conv2d(1, 1, 256, 2, 2,  bias=False, relu=True, bn=True, name='bn4a_branch2a')
         .conv2d(3, 3, 256, 1, 1,  bias=False, relu=True, bn=True, name='bn4a_branch2b')
         .conv2d(1, 1, 1024, 1, 1,  bias=False, relu=False, bn=True, name='bn4a_branch2c'))

        (self.feed('bn4a_branch1',
                   'bn4a_branch2c')
         .add_n(relu=True, name='res4a_relu')
         # ----------div----------
         .conv2d(1, 1, 256, 1, 1,  bias=False, relu=True, bn=True, name='bn4b_branch2a')
         .conv2d(3, 3, 256, 1, 1,  bias=False, relu=True, bn=True, name='bn4b_branch2b')
         .conv2d(1, 1, 1024, 1, 1,  bias=False, relu=False, bn=True, name='bn4b_branch2c'))

        (self.feed('res4a_relu',
                   'bn4b_branch2c')
         .add_n(relu=True, name='res4b_relu')
         # ----------div----------
         .conv2d(1, 1, 256, 1, 1,  bias=False, relu=True, bn=True, name='bn4c_branch2a')
         .conv2d(3, 3, 256, 1, 1,  bias=False, relu=True, bn=True, name='bn4c_branch2b')
         .conv2d(1, 1, 1024, 1, 1,  bias=False, relu=False, bn=True, name='bn4c_branch2c'))

        (self.feed('res4b_relu',
                   'bn4c_branch2c')
         .add_n(relu=True, name='res4c_relu')
         # ----------div----------
         .conv2d(1, 1, 256, 1, 1,  bias=False, relu=True, bn=True, name='bn4d_branch2a')
         .conv2d(3, 3, 256, 1, 1,  bias=False, relu=True, bn=True, name='bn4d_branch2b')
         .conv2d(1, 1, 1024, 1, 1,  bias=False, relu=False, bn=True, name='bn4d_branch2c'))

        (self.feed('res4c_relu',
                   'bn4d_branch2c')
         .add_n(relu=True, name='res4d_relu')
         # ----------div----------
         .conv2d(1, 1, 256, 1, 1,  bias=False, relu=True, bn=True, name='bn4e_branch2a')
         .conv2d(3, 3, 256, 1, 1,  bias=False, relu=True, bn=True, name='bn4e_branch2b')
         .conv2d(1, 1, 1024, 1, 1,  bias=False, relu=False, bn=True, name='bn4e_branch2c'))

        (self.feed('res4d_relu',
                   'bn4e_branch2c')
         .add_n(relu=True, name='res4e_relu')
         # ----------div----------
         .conv2d(1, 1, 256, 1, 1,  bias=False, relu=True, bn=True, name='bn4f_branch2a')
         .conv2d(3, 3, 256, 1, 1,  bias=False, relu=True, bn=True, name='bn4f_branch2b')
         .conv2d(1, 1, 1024, 1, 1,  bias=False, relu=False, bn=True, name='bn4f_branch2c'))

        (self.feed('res4e_relu',
                   'bn4f_branch2c')
         .add_n(relu=True, name='res4f_relu')
         # ----------div----------
         .conv2d(1, 1, 2048, 2, 2,  bias=False, relu=False, bn=True, name='bn5a_branch1'))

        (self.feed('res4f_relu')
         .conv2d(1, 1, 512, 2, 2,  bias=False, relu=True, bn=True, name='bn5a_branch2a')
         .conv2d(3, 3, 512, 1, 1,  bias=False, relu=True, bn=True, name='bn5a_branch2b')
         .conv2d(1, 1, 2048, 1, 1,  bias=False, relu=False, bn=True, name='bn5a_branch2c'))

        (self.feed('bn5a_branch1',
                   'bn5a_branch2c')
         .add_n(relu=True, name='res5a_relu')
         # ----------div----------
         .conv2d(1, 1, 512, 1, 1,  bias=False, relu=True, bn=True, name='bn5b_branch2a')
         .conv2d(3, 3, 512, 1, 1,  bias=False, relu=True, bn=True, name='bn5b_branch2b')
         .conv2d(1, 1, 2048, 1, 1,  bias=False, relu=False, bn=True, name='bn5b_branch2c'))

        (self.feed('res5a_relu',
                   'bn5b_branch2c')
         .add_n(relu=True, name='res5b_relu')
         # ----------div----------
         .conv2d(1, 1, 512, 1, 1,  bias=False, relu=True, bn=True, name='bn5c_branch2a')
         .conv2d(3, 3, 512, 1, 1,  bias=False, relu=True, bn=True, name='bn5c_branch2b')
         .conv2d(1, 1, 2048, 1, 1,  bias=False, relu=False, bn=True, name='bn5c_branch2c'))

        (self.feed('res5b_relu',
                   'bn5c_branch2c')
         .add_n(relu=True, name='res5c_relu')
         .avg_pool(7, 7, 1, 1, padding='VALID', name='avg_pool')
         .fc(out_num=cfg.num_classes, flatten=True, relu=False, name="fc")
         .softmax(name="softmax"))
