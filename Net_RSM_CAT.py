import functools
import torch.nn as nn
import modules_RSM_CAT as ms


class Net(nn.Module):

    def __init__(self, in_nc=3, out_nc=3, nf=64):
        super(Net, self).__init__()

        self.pre_ini = nn.Sequential(nn.Conv2d(in_nc, nf, 3, 1, 1),
                                     nn.LeakyReLU(0.1, True), nn.Conv2d(nf, nf, 3, 1, 1),
                                     nn.LeakyReLU(0.1, True), nn.Conv2d(nf, nf, 1))
        self.PreNet1 = nn.Sequential(nn.Conv2d(nf, nf, 1),
                                     nn.LeakyReLU(0.1, True), nn.Conv2d(nf, 32, 1))
        self.PreNet2 = nn.Sequential(nn.Conv2d(nf, nf, 3, 2, 1),
                                     nn.LeakyReLU(0.1, True), nn.Conv2d(nf, 32, 1))
        self.PreNet3 = nn.Sequential(nn.Conv2d(nf, nf, 3, 2, 1),
                                     nn.LeakyReLU(0.1, True), nn.Conv2d(nf, 32, 3, 2, 1))

        self.conv_ini = nn.Sequential(nn.Conv2d(in_nc, nf, 3, 1, 1),
                                        nn.LeakyReLU(0.1, True))
        
        self.RSM_layer1 = ms.RSM()
        self.conv1 = nn.Sequential(nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
                                        nn.LeakyReLU(0.1, True))

        self.down_conv1 = nn.Sequential(nn.Conv2d(nf, nf, 3, 2, 1),
                                        nn.LeakyReLU(0.1, True))
        self.down_conv2 = nn.Sequential(nn.Conv2d(nf, nf, 3, 2, 1),
                                        nn.LeakyReLU(0.1, True))
        
        basic_block = functools.partial(ms.RSM, n_feat=nf)
        self.MRSM1 = ms.make_layer(basic_block, 2)
        self.MRSM2 = ms.make_layer(basic_block, 8)
        self.MRSM3 = ms.make_layer(basic_block, 2)

        self.cat1 = ms.CAT(image_size=64, patchSize=4)
        self.cat2 = ms.CAT(image_size=128, patchSize=8)
        self.cat3 = ms.CAT(image_size=256, patchSize=16)

        self.up_conv1 = nn.Sequential(nn.Conv2d(nf, nf*4, 3, 1, 1), nn.PixelShuffle(2),
                                      nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.up_conv2 = nn.Sequential(nn.Conv2d(nf, nf*4, 3, 1, 1), nn.PixelShuffle(2),
                                      nn.LeakyReLU(negative_slope=0.1, inplace=True))

        self.RSM_layer2 = ms.RSM()
        self.conv2 = nn.Sequential(nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
                                      nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.mask_est = nn.Conv2d(nf, out_nc, 1)

    def forward(self, x):

        pre = self.pre_ini(x)# 64
        pre1 = self.PreNet1(pre)# 32
        pre2 = self.PreNet2(pre)
        pre3 = self.PreNet3(pre)

        fea0 = self.conv_ini(x)
        fea0,_ = self.RSM_layer1((fea0, pre1))
        fea0 = self.conv1(fea0)

        fea1 = self.down_conv1(fea0)#200
        fea1,_ = self.MRSM1((fea1, pre2))

        fea2 = self.down_conv2(fea1)#100
        out,_ = self.MRSM2((fea2, pre3))
        out = out + self.cat1(fea2)

        out = self.up_conv1(out) + self.cat2(fea1)
        out,_ = self.MRSM3((out, pre2))

        out = self.up_conv2(out) + self.cat3(fea0)
        out,_ = self.RSM_layer2((out, pre1))
        out = self.conv2(out)

        out = self.mask_est(pre) * x + self.conv_last(out)
        return out