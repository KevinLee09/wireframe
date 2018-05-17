import importlib
import pdb
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable, gradcheck
from torch.autograd.gradcheck import gradgradcheck

from model.networks.inception_v2 import inception_v2


class inception(nn.Module):
    def __init__(self, classes, opt):
        super(inception, self).__init__()
        self.classes = classes
        self.n_classes = 2
        self.base_net = inception_v2(num_classes=2, with_bn=opt.hype.get('batchnorm', True))

        decoder_module = importlib.import_module('model.networks.{}_decoder'.format(opt.decoder))
        self.decoder_ = decoder_module.DecodeNet(opt, 'train')

    def forward(self, im_data, junc_conf, junc_res, bin_conf, bin_res):
        # junc_conf, junc_res, bin_conf, bin_res
        base_feat = self.base_net(im_data)
        preds = self.decoder_(base_feat)
        return preds
