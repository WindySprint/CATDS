import torch
import torch.nn as nn
from CTrans import ChannelTransformer

def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)

def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)

def position(H, W, is_cuda=True):
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc

##########################################################################

# 利用1*1卷积得到的三种特征进入交叉transformer，然后把交叉的通过全连接合起来，合起来之前乘以可学习的参数

### --------- Cross-Aggeration Transformer (CAT) ----------
class CAT(nn.Module):
    def __init__(self, in_planes=64, out_planes=64, image_size=256, patchSize=16, head=4, kernel_conv=3):
        super(CAT, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.head = head
        self.kernel_conv = kernel_conv
        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))
        self.rate3 = torch.nn.Parameter(torch.Tensor(1))
        self.head_dim = self.out_planes // self.head

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)

        self.ct = ChannelTransformer(img_size=image_size, patchSize=patchSize, channel_num=64)

        self.fc = nn.Conv2d(3 * self.head, self.kernel_conv * self.kernel_conv, kernel_size=1, bias=False)
        self.dep_conv = nn.Conv2d(self.kernel_conv * self.kernel_conv * self.head_dim, out_planes,
                                  kernel_size=self.kernel_conv, bias=True, groups=self.head_dim, padding=1,
                                  stride=1)

        self.reset_parameters()

    def reset_parameters(self):
        init_rate_half(self.rate1)
        init_rate_half(self.rate2)
        init_rate_half(self.rate3)

        kernel = torch.zeros(self.kernel_conv * self.kernel_conv, self.kernel_conv, self.kernel_conv)
        for i in range(self.kernel_conv * self.kernel_conv):
            kernel[i, i // self.kernel_conv, i % self.kernel_conv] = 1.
        kernel = kernel.squeeze(0).repeat(self.out_planes, 1, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=True)
        self.dep_conv.bias = init_rate_0(self.dep_conv.bias)

    def forward(self, x):
        c1, c2, c3 = self.conv1(x), self.conv2(x), self.conv3(x)
        b, c, h, w = c1.shape
        c1, c2, c3, _ = self.ct(c1, c2, c3)
        ## conv
        f_all = self.fc(torch.cat(
            [self.rate1 * c1.view(b, self.head, self.head_dim, h * w),
             self.rate2 * c2.view(b, self.head, self.head_dim, h * w),
             self.rate3 * c3.view(b, self.head, self.head_dim, h * w)], 1))
        f_conv = f_all.permute(0, 2, 1, 3).reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])
        out_conv = self.dep_conv(f_conv)

        return out_conv

# multi-modules
def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

##########################################################################
### --------- Residual Supplement Module (RSM) ----------
class RSM(nn.Module):
    def __init__(self, n_feat=64):
        super(RSM, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=False, groups=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=False, groups=1)
        )

        self.act = nn.LeakyReLU(0.2)

        self.conv_mask = nn.Sequential(
            nn.Conv2d(n_feat//2, n_feat, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_feat, 1, kernel_size=1, bias=False)
        )

        self.softmax = nn.Softmax(dim=2)

        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=False)
        )

        self.conv_add = nn.Sequential(
            nn.Conv2d(n_feat//2, n_feat, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=False)
        )

    def forward(self, x):
        # original input 64:32
        input_x = self.body(x[0])
        b, c, h, w = input_x.size()
        # [B, C, H * W]
        input_x = input_x.view(b, c, h * w)
        # [B, 1, C, H * W]
        input_x = input_x.unsqueeze(1)

        # supplement
        # [B, 1, H, W]
        supplement_mask = self.conv_mask(x[1])
        # [B, 1, H * W]
        supplement_mask = supplement_mask.view(b, 1, h * w)
        # [B, 1, H * W]
        supplement_mask = self.softmax(supplement_mask)
        # [B, 1, H * W, 1]
        supplement_mask = supplement_mask.unsqueeze(3)

        # matmul
        # [B, 1, C, 1]
        context = torch.matmul(input_x, supplement_mask)
        # [B, C, 1, 1]
        context = context.view(b, c, 1, 1)

        # add
        # [B, C, 1, 1]
        channel_add_term = self.channel_add_conv(context)
        supplement_add = self.conv_add(x[1])
        sc_add = supplement_add + channel_add_term

        sc_add = self.act(sc_add)
        sc_add += x[0]
        return (sc_add, x[1])
