import mxm.release.res_net_101.config as cfg
from mxm.release.base_symble import BaseSymble


class Symble(BaseSymble):
    def def_model(self):
        (self.feed('data')
         .conv2d(7, 7, 64, 2, 2, bias=False, relu=False, bn=True, name='conv1')
         .max_pool(3, 3, 2, 2, name='pool1')
         .conv2d(1, 1, 256, 1, 1, bias=False, relu=False, bn=True, name='bn2a_branch1'))

        (self.feed('pool1')
         .conv2d(1, 1, 64, 1, 1, bias=False, relu=False, bn=True, name='res2a_branch2a')
         .conv2d(3, 3, 64, 1, 1, bias=False, relu=False, bn=True, name='res2a_branch2b')
         .conv2d(1, 1, 256, 1, 1, bias=False, relu=False, bn=True, name='bn2a_branch2c'))

        # flag =====================
        (self.feed('bn2a_branch1',
                   'bn2a_branch2c')
         .add_n(relu=True, name='res2a_relu')
         .conv2d(1, 1, 64, 1, 1, bias=False, relu=False, bn=True, name='res2b_branch2a')
         .conv2d(3, 3, 64, 1, 1, bias=False, relu=False, bn=True, name='res2b_branch2b')
         .conv2d(1, 1, 256, 1, 1, bias=False, relu=False, bn=True, name='bn2b_branch2c'))

        (self.feed('res2a_relu',
                   'bn2b_branch2c')
         .add_n(relu=True, name='res2b_relu')
         .conv2d(1, 1, 64, 1, 1, bias=False, relu=False, bn=True, name='res2c_branch2a')
         .conv2d(3, 3, 64, 1, 1, bias=False, relu=False, bn=True, name='res2c_branch2b')
         .conv2d(1, 1, 256, 1, 1, bias=False, relu=False, bn=True, name='bn2c_branch2c'))

        (self.feed('res2b_relu',
                   'bn2c_branch2c')
         .add_n(relu=True, name='res2c_relu')
         .conv2d(1, 1, 512, 2, 2, bias=False, relu=False, bn=True, name='bn3a_branch1'))

        (self.feed('res2c_relu')
         .conv2d(1, 1, 128, 2, 2, bias=False, relu=False, bn=True, name='res3a_branch2a')
         .conv2d(3, 3, 128, 1, 1, bias=False, relu=False, bn=True, name='res3a_branch2b')
         .conv2d(1, 1, 512, 1, 1, bias=False, relu=False, bn=True, name='bn3a_branch2c'))

        # flag =====================
        (self.feed('bn3a_branch1',
                   'bn3a_branch2c')
         .add_n(relu=True, name='res3a_relu')
         .conv2d(1, 1, 128, 1, 1, bias=False, relu=False, bn=True, name='res3b1_branch2a')
         .conv2d(3, 3, 128, 1, 1, bias=False, relu=False, bn=True, name='res3b1_branch2b')
         .conv2d(1, 1, 512, 1, 1, bias=False, relu=False, bn=True, name='bn3b1_branch2c'))

        (self.feed('res3a_relu',
                   'bn3b1_branch2c')
         .add_n(relu=True, name='res3b1_relu')
         .conv2d(1, 1, 128, 1, 1, bias=False, relu=False, bn=True, name='res3b2_branch2a')
         .conv2d(3, 3, 128, 1, 1, bias=False, relu=False, bn=True, name='res3b2_branch2b')
         .conv2d(1, 1, 512, 1, 1, bias=False, relu=False, bn=True, name='bn3b2_branch2c'))

        (self.feed('res3b1_relu',
                   'bn3b2_branch2c')
         .add_n(relu=True, name='res3b2_relu')
         .conv2d(1, 1, 128, 1, 1, bias=False, relu=False, bn=True, name='res3b3_branch2a')
         .conv2d(3, 3, 128, 1, 1, bias=False, relu=False, bn=True, name='res3b3_branch2b')
         .conv2d(1, 1, 512, 1, 1, bias=False, relu=False, bn=True, name='bn3b3_branch2c'))

        (self.feed('res3b2_relu',
                   'bn3b3_branch2c')
         .add_n(relu=True, name='res3b3_relu')
         .conv2d(1, 1, 1024, 2, 2, bias=False, relu=False, bn=True, name='bn4a_branch1'))

        (self.feed('res3b3_relu')
         .conv2d(1, 1, 256, 2, 2, bias=False, relu=False, bn=True, name='res4a_branch2a')
         .conv2d(3, 3, 256, 1, 1, bias=False, relu=False, bn=True, name='res4a_branch2b')
         .conv2d(1, 1, 1024, 1, 1, bias=False, relu=False, bn=True, name='bn4a_branch2c'))

        # flag =====================
        (self.feed('bn4a_branch1',
                   'bn4a_branch2c')
         .add_n(relu=True, name='res4a_relu')
         .conv2d(1, 1, 256, 1, 1, bias=False, relu=False, bn=True, name='res4b1_branch2a')
         .conv2d(3, 3, 256, 1, 1, bias=False, relu=False, bn=True, name='res4b1_branch2b')
         .conv2d(1, 1, 1024, 1, 1, bias=False, relu=False, bn=True, name='bn4b1_branch2c'))

        (self.feed('res4a_relu',
                   'bn4b1_branch2c')
         .add_n(relu=True, name='res4b1_relu')
         .conv2d(1, 1, 256, 1, 1, bias=False, relu=False, bn=True, name='res4b2_branch2a')
         .conv2d(3, 3, 256, 1, 1, bias=False, relu=False, bn=True, name='res4b2_branch2b')
         .conv2d(1, 1, 1024, 1, 1, bias=False, relu=False, bn=True, name='bn4b2_branch2c'))

        (self.feed('res4b1_relu',
                   'bn4b2_branch2c')
         .add_n(relu=True, name='res4b2_relu')
         .conv2d(1, 1, 256, 1, 1, bias=False, relu=False, bn=True, name='res4b3_branch2a')
         .conv2d(3, 3, 256, 1, 1, bias=False, relu=False, bn=True, name='res4b3_branch2b')
         .conv2d(1, 1, 1024, 1, 1, bias=False, relu=False, bn=True, name='bn4b3_branch2c'))

        (self.feed('res4b2_relu',
                   'bn4b3_branch2c')
         .add_n(relu=True, name='res4b3_relu')
         .conv2d(1, 1, 256, 1, 1, bias=False, relu=False, bn=True, name='res4b4_branch2a')
         .conv2d(3, 3, 256, 1, 1, bias=False, relu=False, bn=True, name='res4b4_branch2b')
         .conv2d(1, 1, 1024, 1, 1, bias=False, relu=False, bn=True, name='bn4b4_branch2c'))

        (self.feed('res4b3_relu',
                   'bn4b4_branch2c')
         .add_n(relu=True, name='res4b4_relu')
         .conv2d(1, 1, 256, 1, 1, bias=False, relu=False, bn=True, name='res4b5_branch2a')
         .conv2d(3, 3, 256, 1, 1, bias=False, relu=False, bn=True, name='res4b5_branch2b')
         .conv2d(1, 1, 1024, 1, 1, bias=False, relu=False, bn=True, name='bn4b5_branch2c'))

        (self.feed('res4b4_relu',
                   'bn4b5_branch2c')
         .add_n(relu=True, name='res4b5_relu')
         .conv2d(1, 1, 256, 1, 1, bias=False, relu=False, bn=True, name='res4b6_branch2a')
         .conv2d(3, 3, 256, 1, 1, bias=False, relu=False, bn=True, name='res4b6_branch2b')
         .conv2d(1, 1, 1024, 1, 1, bias=False, relu=False, bn=True, name='bn4b6_branch2c'))

        (self.feed('res4b5_relu',
                   'bn4b6_branch2c')
         .add_n(relu=True, name='res4b6_relu')
         .conv2d(1, 1, 256, 1, 1, bias=False, relu=False, bn=True, name='res4b7_branch2a')
         .conv2d(3, 3, 256, 1, 1, bias=False, relu=False, bn=True, name='res4b7_branch2b')
         .conv2d(1, 1, 1024, 1, 1, bias=False, relu=False, bn=True, name='bn4b7_branch2c'))

        (self.feed('res4b6_relu',
                   'bn4b7_branch2c')
         .add_n(relu=True, name='res4b7_relu')
         .conv2d(1, 1, 256, 1, 1, bias=False, relu=False, bn=True, name='res4b8_branch2a')
         .conv2d(3, 3, 256, 1, 1, bias=False, relu=False, bn=True, name='res4b8_branch2b')
         .conv2d(1, 1, 1024, 1, 1, bias=False, relu=False, bn=True, name='bn4b8_branch2c'))

        (self.feed('res4b7_relu',
                   'bn4b8_branch2c')
         .add_n(relu=True, name='res4b8_relu')
         .conv2d(1, 1, 256, 1, 1, bias=False, relu=False, bn=True, name='res4b9_branch2a')
         .conv2d(3, 3, 256, 1, 1, bias=False, relu=False, bn=True, name='res4b9_branch2b')
         .conv2d(1, 1, 1024, 1, 1, bias=False, relu=False, bn=True, name='bn4b9_branch2c'))

        (self.feed('res4b8_relu',
                   'bn4b9_branch2c')
         .add_n(relu=True, name='res4b9_relu')
         .conv2d(1, 1, 256, 1, 1, bias=False, relu=False, bn=True, name='res4b10_branch2a')
         .conv2d(3, 3, 256, 1, 1, bias=False, relu=False, bn=True, name='res4b10_branch2b')
         .conv2d(1, 1, 1024, 1, 1, bias=False, relu=False, bn=True, name='bn4b10_branch2c'))

        (self.feed('res4b9_relu',
                   'bn4b10_branch2c')
         .add_n(relu=True, name='res4b10_relu')
         .conv2d(1, 1, 256, 1, 1, bias=False, relu=False, bn=True, name='res4b11_branch2a')
         .conv2d(3, 3, 256, 1, 1, bias=False, relu=False, bn=True, name='res4b11_branch2b')
         .conv2d(1, 1, 1024, 1, 1, bias=False, relu=False, bn=True, name='bn4b11_branch2c'))

        (self.feed('res4b10_relu',
                   'bn4b11_branch2c')
         .add_n(relu=True, name='res4b11_relu')
         .conv2d(1, 1, 256, 1, 1, bias=False, relu=False, bn=True, name='res4b12_branch2a')
         .conv2d(3, 3, 256, 1, 1, bias=False, relu=False, bn=True, name='res4b12_branch2b')
         .conv2d(1, 1, 1024, 1, 1, bias=False, relu=False, bn=True, name='bn4b12_branch2c'))

        (self.feed('res4b11_relu',
                   'bn4b12_branch2c')
         .add_n(relu=True, name='res4b12_relu')
         .conv2d(1, 1, 256, 1, 1, bias=False, relu=False, bn=True, name='res4b13_branch2a')
         .conv2d(3, 3, 256, 1, 1, bias=False, relu=False, bn=True, name='res4b13_branch2b')
         .conv2d(1, 1, 1024, 1, 1, bias=False, relu=False, bn=True, name='bn4b13_branch2c'))

        (self.feed('res4b12_relu',
                   'bn4b13_branch2c')
         .add_n(relu=True, name='res4b13_relu')
         .conv2d(1, 1, 256, 1, 1, bias=False, relu=False, bn=True, name='res4b14_branch2a')
         .conv2d(3, 3, 256, 1, 1, bias=False, relu=False, bn=True, name='res4b14_branch2b')
         .conv2d(1, 1, 1024, 1, 1, bias=False, relu=False, bn=True, name='bn4b14_branch2c'))

        (self.feed('res4b13_relu',
                   'bn4b14_branch2c')
         .add_n(relu=True, name='res4b14_relu')
         .conv2d(1, 1, 256, 1, 1, bias=False, relu=False, bn=True, name='res4b15_branch2a')
         .conv2d(3, 3, 256, 1, 1, bias=False, relu=False, bn=True, name='res4b15_branch2b')
         .conv2d(1, 1, 1024, 1, 1, bias=False, relu=False, bn=True, name='bn4b15_branch2c'))

        (self.feed('res4b14_relu',
                   'bn4b15_branch2c')
         .add_n(relu=True, name='res4b15_relu')
         .conv2d(1, 1, 256, 1, 1, bias=False, relu=False, bn=True, name='res4b16_branch2a')
         .conv2d(3, 3, 256, 1, 1, bias=False, relu=False, bn=True, name='res4b16_branch2b')
         .conv2d(1, 1, 1024, 1, 1, bias=False, relu=False, bn=True, name='bn4b16_branch2c'))

        (self.feed('res4b15_relu',
                   'bn4b16_branch2c')
         .add_n(relu=True, name='res4b16_relu')
         .conv2d(1, 1, 256, 1, 1, bias=False, relu=False, bn=True, name='res4b17_branch2a')
         .conv2d(3, 3, 256, 1, 1, bias=False, relu=False, bn=True, name='res4b17_branch2b')
         .conv2d(1, 1, 1024, 1, 1, bias=False, relu=False, bn=True, name='bn4b17_branch2c'))

        (self.feed('res4b16_relu',
                   'bn4b17_branch2c')
         .add_n(relu=True, name='res4b17_relu')
         .conv2d(1, 1, 256, 1, 1, bias=False, relu=False, bn=True, name='res4b18_branch2a')
         .conv2d(3, 3, 256, 1, 1, bias=False, relu=False, bn=True, name='res4b18_branch2b')
         .conv2d(1, 1, 1024, 1, 1, bias=False, relu=False, bn=True, name='bn4b18_branch2c'))

        (self.feed('res4b17_relu',
                   'bn4b18_branch2c')
         .add_n(relu=True, name='res4b18_relu')
         .conv2d(1, 1, 256, 1, 1, bias=False, relu=False, bn=True, name='res4b19_branch2a')
         .conv2d(3, 3, 256, 1, 1, bias=False, relu=False, bn=True, name='res4b19_branch2b')
         .conv2d(1, 1, 1024, 1, 1, bias=False, relu=False, bn=True, name='bn4b19_branch2c'))

        (self.feed('res4b18_relu',
                   'bn4b19_branch2c')
         .add_n(relu=True, name='res4b19_relu')
         .conv2d(1, 1, 256, 1, 1, bias=False, relu=False, bn=True, name='res4b20_branch2a')
         .conv2d(3, 3, 256, 1, 1, bias=False, relu=False, bn=True, name='res4b20_branch2b')
         .conv2d(1, 1, 1024, 1, 1, bias=False, relu=False, bn=True, name='bn4b20_branch2c'))

        (self.feed('res4b19_relu',
                   'bn4b20_branch2c')
         .add_n(relu=True, name='res4b20_relu')
         .conv2d(1, 1, 256, 1, 1, bias=False, relu=False, bn=True, name='res4b21_branch2a')
         .conv2d(3, 3, 256, 1, 1, bias=False, relu=False, bn=True, name='res4b21_branch2b')
         .conv2d(1, 1, 1024, 1, 1, bias=False, relu=False, bn=True, name='bn4b21_branch2c'))

        (self.feed('res4b20_relu',
                   'bn4b21_branch2c')
         .add_n(relu=True, name='res4b21_relu')
         .conv2d(1, 1, 256, 1, 1, bias=False, relu=False, bn=True, name='res4b22_branch2a')
         .conv2d(3, 3, 256, 1, 1, bias=False, relu=False, bn=True, name='res4b22_branch2b')
         .conv2d(1, 1, 1024, 1, 1, bias=False, relu=False, bn=True, name='bn4b22_branch2c'))

        (self.feed('res4b21_relu',
                   'bn4b22_branch2c')
         .add_n(relu=True, name='res4b22_relu')
         .conv2d(1, 1, 2048, 2, 2, bias=False, relu=False, bn=True, name='bn5a_branch1'))

        (self.feed('res4b22_relu')
         .conv2d(1, 1, 512, 2, 2, bias=False, relu=False, bn=True, name='res5a_branch2a')
         .conv2d(3, 3, 512, 1, 1, bias=False, relu=False, bn=True, name='res5a_branch2b')
         .conv2d(1, 1, 2048, 1, 1, bias=False, relu=False, bn=True, name='bn5a_branch2c'))

        # flag =====================
        (self.feed('bn5a_branch1',
                   'bn5a_branch2c')
         .add_n(relu=True, name='res5a_relu')
         .conv2d(1, 1, 512, 1, 1, bias=False, relu=False, bn=True, name='res5b_branch2a')
         .conv2d(3, 3, 512, 1, 1, bias=False, relu=False, bn=True, name='res5b_branch2b')
         .conv2d(1, 1, 2048, 1, 1, bias=False, relu=False, bn=True, name='bn5b_branch2c'))

        (self.feed('res5a_relu',
                   'bn5b_branch2c')
         .add_n(relu=True, name='res5b_relu')
         .conv2d(1, 1, 512, 1, 1, bias=False, relu=False, bn=True, name='res5c_branch2a')
         .conv2d(3, 3, 512, 1, 1, bias=False, relu=False, bn=True, name='res5c_branch2b')
         .conv2d(1, 1, 2048, 1, 1, bias=False, relu=False, bn=True, name='bn5c_branch2c'))

        (self.feed('res5b_relu',
                   'bn5c_branch2c')
         .add_n(relu=True, name='res5c_relu')
         .avg_pool(7, 7, 1, 1, padding='VALID', name='avg_pool')
         .fc(out_num=cfg.num_classes, flatten=True, relu=False, name="fc")
         .softmax(name="softmax"))
