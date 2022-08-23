#!/usr/bin/env python3
# coding: utf-8
import argparse
import os
import numpy as np
import time
import cv2

from modeling.unet import *
from dataloaders import custom_transforms as tr
from PIL import Image
from torchvision import transforms
from dataloaders.utils import  *
from torchvision.utils import make_grid, save_image

def main():

    pthfile = 'run/pascal/Unet/model_best.pth.tar'
    onnxpath = 'run/pascal/Unet/model_best.onnx'


    parser = argparse.ArgumentParser(description="PyTorch Unet Test")

    parser.add_argument('--ckpt', type=str, default='model_best.pth.tar',
                        help='saved model')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')


    parser.add_argument('--num_classes', type=int, default=8,
                        help='crop image size')

    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')


    model = Unet(n_channels=3, n_classes=5)

    batch_size = 1
    input_shape = (3,640,512)

    ckpt = torch.load(args.ckpt)
    model.load_state_dict(ckpt['state_dict'])
    model = model.cuda()

    model.eval()

    x = torch.rand(batch_size, *input_shape)  # 生成张量
    input_names = ['input']
    output_names = ['output']

    torch.onnx.export(model,
                      x,
                      onnxpath,
                      export_params=True,
                      opset_version=12,
                      do_constant_folding=True,  # 是否执行常量折叠优化
                      input_names=['input'],  # 输入名
                      output_names=['output'],  # 输出名
                      dynamic_axes={'input': {0: 'batch_size'},  # 批处理变量
                                    'output': {0: 'batch_size'}}
                      )

if __name__ == "__main__":
   main()
