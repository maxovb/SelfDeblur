
from __future__ import print_function
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
from networks.skip import skip
from networks.fcn import fcn
import cv2
import torch
import torch.optim
import glob
import math
from skimage.io import imread
from skimage.io import imsave
from skimage.metrics import peak_signal_noise_ratio
import warnings
import time
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from utils.common_utils import *
from utils.metrics import comparison_up_to_shift
from SSIM import SSIM

parser = argparse.ArgumentParser()
parser.add_argument('--num_iter', type=int, default=5000, help='number of epochs of training')
parser.add_argument('--img_size', type=int, default=[256, 256], help='size of each image dimension')
parser.add_argument('--kernel_size', type=int, default=[21, 21], help='size of blur kernel [height, width]')
parser.add_argument('--downsize',type=int,default=1,help='Ratio to downsize the image to enable faster debugging')
parser.add_argument('--data_path', type=str, default="datasets/levin/", help='path to blurry image')
parser.add_argument('--save_path', type=str, default="results/levin/", help='path to save results')
parser.add_argument('--save_frequency', type=int, default=100, help='frequency to save results')
parser.add_argument('--loss_frequency', type=int, default=100, help='frequency to compute the losses to the gt image')
opt = parser.parse_args()
#print(opt)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

warnings.filterwarnings("ignore")

files_source = glob.glob(os.path.join(opt.data_path, '*.png'))
files_source.sort()
save_path = opt.save_path
os.makedirs(save_path, exist_ok=True)

# start #image
for f in files_source:
    INPUT = 'noise'
    pad = 'reflection'
    LR = 0.01
    num_iter = opt.num_iter
    reg_noise_std = 0.001

    path_to_image = f
    imgname = os.path.basename(f)
    imgname = os.path.splitext(imgname)[0]

    if imgname.find('kernel1') != -1:
        opt.kernel_size = [17, 17]
    if imgname.find('kernel2') != -1:
        opt.kernel_size = [15, 15]
    if imgname.find('kernel3') != -1:
        opt.kernel_size = [13, 13]
    if imgname.find('kernel4') != -1:
        opt.kernel_size = [27, 27]
    if imgname.find('kernel5') != -1:
        opt.kernel_size = [11, 11]
    if imgname.find('kernel6') != -1:
        opt.kernel_size = [19, 19]
    if imgname.find('kernel7') != -1:
        opt.kernel_size = [21, 21]
    if imgname.find('kernel8') != -1:
        opt.kernel_size = [21, 21]

    _, imgs = get_image(path_to_image, -1) # load image and convert to np.
    y = np_to_torch(imgs).type(dtype)

    # get the real image
    path_to_gt_image = os.path.join(opt.data_path,"gt/im"+ path_to_image.split("im")[1][0] + ".png")
    _, gt_imgs = get_image(path_to_gt_image, -1)  # load image and convert to np.
    x_gt = np_to_torch(gt_imgs).type(dtype)

    # downsize the image
    if opt.downsize != 1:
        opt.kernel_size = [math.ceil(x/opt.downsize) for x in opt.kernel_size]
        y = torchvision.transforms.Resize([y.shape[-2]//opt.downsize,y.shape[-1]//opt.downsize])(y)
        x_gt = torchvision.transforms.Resize([x_gt.shape[-2] // opt.downsize, x_gt.shape[-1] // opt.downsize])(y)

    img_size = y.shape[-3:]
    print(imgname)
    # ######################################################################
    padh, padw = opt.kernel_size[0]-1, opt.kernel_size[1]-1
    opt.img_size[0], opt.img_size[1] = img_size[1]+padh, img_size[2]+padw

    '''
    x_net:
    '''
    input_depth = 8

    net_input = get_noise(input_depth, INPUT, (opt.img_size[0], opt.img_size[1])).type(dtype)

    net = skip( input_depth, 1,
                num_channels_down = [128, 128, 128, 128, 128],
                num_channels_up   = [128, 128, 128, 128, 128],
                num_channels_skip = [16, 16, 16, 16, 16],
                upsample_mode='bilinear',
                need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

    net = net.type(dtype)

    '''
    k_net:
    '''
    n_k = 200
    net_input_kernel = get_noise(n_k, INPUT, (1, 1)).type(dtype)
    net_input_kernel.squeeze_()

    net_kernel = fcn(n_k, opt.kernel_size[0]*opt.kernel_size[1])
    net_kernel = net_kernel.type(dtype)

    # Losses
    mse = torch.nn.MSELoss().type(dtype)
    ssim = SSIM().type(dtype)

    # optimizer
    optimizer = torch.optim.Adam([{'params':net.parameters()},{'params':net_kernel.parameters(),'lr':1e-4}], lr=LR)
    scheduler = MultiStepLR(optimizer, milestones=[2000, 3000, 4000], gamma=0.5)  # learning rates

    # initilization inputs
    net_input_saved = net_input.detach().clone()
    net_input_kernel_saved = net_input_kernel.detach().clone()

    # initialization list of losses
    psnr_gt_list = []
    ssim_gt_list = []
    losses_step = []

    ### start SelfDeblur
    for step in tqdm(range(num_iter), position=0, leave=True):

        # input regularization
        net_input = net_input_saved + reg_noise_std * torch.zeros(net_input_saved.shape).type_as(net_input_saved.data).normal_()

        # change the learning rate
        scheduler.step(step)
        optimizer.zero_grad()

        # get the network output
        out_x = net(net_input)
        out_k = net_kernel(net_input_kernel)
    
        out_k_m = out_k.view(-1,1,opt.kernel_size[0],opt.kernel_size[1])
        # print(out_k_m)
        out_y = nn.functional.conv2d(out_x, out_k_m, padding=0, bias=None)

        if step < 1000:
            total_loss = mse(out_y,y) 
        else:
            total_loss = 1-ssim(out_y, y) 

        total_loss.backward()
        optimizer.step()

        if (step+1) % opt.loss_frequency == 0:
            # compute the losses to the gt image
            out_x_cropped = out_x[:, :, padh // 2:padh // 2 + img_size[1], padw // 2:padw // 2 + img_size[2]]
            ssim_gt, psnr_gt = comparison_up_to_shift(x_gt.detach().cpu().numpy()[0, 0],
                                                      out_x_cropped.detach().cpu().numpy()[0, 0], maxshift=5)
            # store the losses to the gt image
            psnr_gt_list.append(psnr_gt)
            ssim_gt_list.append(ssim_gt)
            losses_step.append(step+1)

        if (step+1) % opt.save_frequency == 0:
            #print('Iteration %05d' %(step+1))

            save_path = os.path.join(opt.save_path, '%s_x.png'%imgname)
            out_x_np = torch_to_np(out_x)
            out_x_np = out_x_np.squeeze()
            out_x_np = out_x_np[padh//2:padh//2+img_size[1], padw//2:padw//2+img_size[2]]
            imsave(save_path, out_x_np)

            save_path = os.path.join(opt.save_path, '%s_k.png'%imgname)
            out_k_np = torch_to_np(out_k_m)
            out_k_np = out_k_np.squeeze()
            out_k_np /= np.max(out_k_np)
            imsave(save_path, out_k_np)

            torch.save(net, os.path.join(opt.save_path, "%s_xnet.pth" % imgname))
            torch.save(net_kernel, os.path.join(opt.save_path, "%s_knet.pth" % imgname))

            save_path = os.path.join(opt.save_path, '%s_SSIM_gt.png' % imgname)
            plt.figure()
            plt.plot(losses_step,ssim_gt_list)
            plt.xlabel('Iterations', fontsize=15)
            plt.ylabel('SSIM', fontsize=15)
            plt.savefig(save_path)
            plt.close()

            save_path = os.path.join(opt.save_path, '%s_PSNR_gt.png' % imgname)
            plt.figure()
            plt.plot(losses_step,psnr_gt_list)
            plt.xlabel('Iterations', fontsize=15)
            plt.ylabel('PSNR', fontsize=15)
            plt.savefig(save_path)
            plt.close()
