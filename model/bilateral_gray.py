#!/usr/bin/python
# torch_bilateral: bi/trilateral filtering in torch
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np
import pdb
import time


def gkern2d(l=21, sig=3):
    """Returns a 2D Gaussian kernel array."""
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sig ** 2))
    return kernel


class Shift(nn.Module):
    def __init__(self, in_planes, kernel_size=3):
        super(Shift, self).__init__()
        self.in_planes = in_planes
        self.kernel_size = kernel_size
        self.channels_per_group = self.in_planes // (self.kernel_size ** 2)
        if self.kernel_size == 3:
            self.pad = 1
        elif self.kernel_size == 5:
            self.pad = 2
        elif self.kernel_size == 7:
            self.pad = 3

    def forward(self, x):
        n, c, h, w = x.size()
        x_pad = F.pad(x, (self.pad, self.pad, self.pad, self.pad))
        # Alias for convenience
        cpg = self.channels_per_group
        cat_layers = []
        #Parse in row-major
        for y in range(0,self.kernel_size):
            y2 = y+h
            for x in range(0, self.kernel_size):
                x2 = x+w
                xx = x_pad[:,:,y:y2,x:x2]
                cat_layers += [xx]
        return torch.cat(cat_layers, 1)


class BilateralFilter(nn.Module):
    r"""BilateralFilter computes:
        If = 1/W * Sum_{xi C Omega}(I * f(||I(xi)-I(x)||) * g(||xi-x||))
    """

    def __init__(self, channels=3, k=7, height=480, width=640, sigma_space=5, sigma_color=0.1):
        super(BilateralFilter, self).__init__()

        #space gaussian kernel
        #FIXME: do everything in torch
        self.g = Parameter(torch.Tensor(channels,k*k), requires_grad = False)
        self.gw = gkern2d(k,sigma_space)

        gw = np.tile(self.gw.reshape(1,k*k,1,1),(1,1,height,width))
        self.g.data = torch.from_numpy(gw).float()
        #shift
        self.shift = Shift(channels,k)
        self.sigma_color = 2*sigma_color**2

    def forward(self, I):
        Is = self.shift(I).data
        Iex = I.expand(*Is.size())
        D = (Is-Iex)**2 #here we are actually missing some sum over groups of channels
        De = torch.exp(-D / self.sigma_color)
        Dd = De * self.g.data
        W_denom = torch.sum(Dd,dim=1)
        If = torch.sum(Dd*Is,dim=1) / W_denom
        return If




if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import cv2

    c,h,w = 1,480/2,640/2
    k = 5
    cuda = False

    bilat = BilateralFilter(c,k,h,w)
    if cuda:
        bilat.cuda()


    im = cv2.imread('/home/eperot/Pictures/Lena.png', cv2.IMREAD_GRAYSCALE)
    im = cv2.resize(im,(w,h),interpolation=cv2.INTER_CUBIC)
    im_in = im.reshape(1,1,h,w)
    img = torch.from_numpy(im_in).float() / 255.0

    if cuda:
        img_in = img.cuda()
    else:
        img_in = img

    y = bilat(img_in)

    start = time.time()
    y = bilat(img_in)
    print(time.time()-start)


    img_out = y.cpu().numpy()[0] #not counting the return transfer in timing!



    show_out = cv2.resize(img_out,(640,480))
    show_in = cv2.resize(img[0,0].numpy(),(640,480))

    # diff = np.abs(img_out - img[0,0])
    # diff = (diff - diff.min()) / (diff.max() - diff.min())
    # cv2.namedWindow('diff')
    # cv2.moveWindow('diff',50,50)
    # cv2.imshow('diff', diff)



    cv2.namedWindow('kernel')
    cv2.imshow('kernel', bilat.gw)


    cv2.namedWindow('img_out')
    cv2.moveWindow('img_out', 200, 200)
    cv2.imshow('img_out',show_out)
    #
    cv2.namedWindow('img_in')
    cv2.imshow('img_in', show_in)

    cv2.waitKey(0)


    # n = gkern2d(5,3)
    #
    # s = n.reshape(1,25)
    #
    # print(n)
    # print(s)
    # plt.imshow(n, interpolation='none')
    # plt.show()
