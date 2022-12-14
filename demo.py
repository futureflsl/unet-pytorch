
#
# demo.py
#
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

#np.set_printoptions(threshold=np.inf)

def main():

    path_file=r'/home/fut/Downloads/UNet/mydata/images'

    parser = argparse.ArgumentParser(description="PyTorch Unet Test")
    parser.add_argument('--in-path',type=str,default=path_file,help='image to test')
    #parser.add_argument('--in-path', type=str, required=True, help='image to test')
    parser.add_argument('--ckpt', type=str, default='run/pascal/Unet/model_best.pth.tar',
                        help='saved model')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes','invoice'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--crop-size', type=int, default=512,
                        help='crop image size')
    parser.add_argument('--num_classes', type=int, default=5,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False
    model_s_time = time.time()
    model = Unet(n_channels=3, n_classes=8)

    ckpt = torch.load(args.ckpt, map_location='cpu')

    model.load_state_dict(ckpt['state_dict'])



    model = model.cuda()
    model_u_time = time.time()
    model_load_time = model_u_time-model_s_time

    print("model load time is {}".format(model_load_time))

    composed_transforms = transforms.Compose([
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()])



    for name_i in os.listdir(path_file):


        s_time = time.time()

        image = Image.open(args.in_path+"/"+name_i).convert('RGB')
        target = Image.open(args.in_path+"/"+name_i).convert('L')
        sample = {'image': image, 'label': target}

        tensor_in = composed_transforms(sample)['image'].unsqueeze(0)

        model.eval()
        if args.cuda:
            tensor_in = tensor_in.cuda()
        with torch.no_grad():
            output = model(tensor_in)     #????????????

        grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy()),
                            3, normalize=False, range=(0, 9))

        '''
        ???????????????
        '''

        # ??????????????????????????????????????????????????????????????????????????????

        save_image(grid_image,"./run/{}.jpg".format(name_i[0:-4]))

        u_time = time.time()
        img_time = u_time - s_time
        print("image:{} time: {} ".format(name_i, img_time))


    print("image save in in_path.")

if __name__ == "__main__":
   main()

# python demo.py --in-path your_file --out-path your_dst_file

