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
import math
import glob
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio
import warnings
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from utils.common_utils import *
from utils.metrics import comparison_up_to_shift
from SSIM import SSIM
from utils.training_utils import add_noise_weights_model, add_noise_gradients_model, backtracking
from utils.SGLD import SGLD, pSGLD

parser = argparse.ArgumentParser()
parser.add_argument('--num_iter', type=int, default=20000, help='Number of epochs of training. Default: 20000.')
parser.add_argument('--img_size', type=int, default=[256, 256], help='Size of each image dimension. Default: [256,256].')
parser.add_argument('--downsize',type=int,default=1,help='Ratio to downsize the image to enable faster debugging')
parser.add_argument('--kernel_size', type=int, default=[21, 21], help='Size of blur kernel [height, width]. Default: [21,21].')
parser.add_argument('--data_path', type=str, default="datasets/levin/", help='Path to blurry image. Default: datasets/levin/ .')
parser.add_argument('--save_path', type=str, default="results/GP_levin_JASGLD/", help='Path to save results. Default: results/GP_levin/ .')
parser.add_argument('--learning_rate', type=bool, default=0.01, help='Learnign rate to scale the gradient update steps. Default: 0.01.')
parser.add_argument('--save_frequency', type=int, default=100, help='Frequency to save results. Defaults: 100.')
parser.add_argument('--loss_frequency', type=int, default=100, help='Frequency to compute the losses to the gt image. Defaults: 100.')
parser.add_argument('--use_preconditioning', type=bool, nargs='?', action='store', default=True, help='Use RMSprop preconditioning. Default: True.')
parser.add_argument('--weight_decay',type=float,default=5e-8, help='Weight decay to penalize large weights. Default: 5e-8.')
parser.add_argument('--averaging_iter',type=int,default=51, help='Number of iterations over which local MCMC are averaged. Default: 51.')
parser.add_argument('--MCMC_iter',type=int,default=500, help='Number of iterations for sampling for the global MCMC. Default: 500.')
parser.add_argument('--burnin_iter',type=int,default=7000, help='Number of iterations after which to start sampling. Default: 7000.')
parser.add_argument('--roll_back',type = bool, nargs = '?', action = 'store', default = True, help='Use backrolling if large drop in PSNR. Default: True.')
opt = parser.parse_args()

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

roll_back = opt.roll_back

# start #image
for f in files_source:
    INPUT = 'noise'
    pad = 'reflection'
    LR = opt.learning_rate
    num_iter = opt.num_iter
    reg_noise_std = 0.001
    weight_decay = opt.weight_decay

    # parameters for the SGLD
    averaging_iter = opt.averaging_iter
    MCMC_iter = opt.MCMC_iter
    burnin_iter = opt.burnin_iter

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
    path_to_gt_image = os.path.join(opt.data_path, "gt/im" + path_to_image.split("im")[1][0] + ".png")
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
    if opt.use_preconditioning:
        optimizer = pSGLD([{'params':net.parameters()},{'params':net_kernel.parameters(),'lr':1e-4}], lr=LR, norm_sigma= 1/math.sqrt(weight_decay), addnoise=True)
    else:
        optimizer = SGLD([{'params':net.parameters()},{'params':net_kernel.parameters(),'lr':1e-4}], lr=LR, norm_sigma= 1/math.sqrt(weight_decay), addnoise=True)
    lambda1 = lambda step: 1/(step+1)
    scheduler = LambdaLR(optimizer,lr_lambda=lambda1)

    # initilization inputs
    net_input_saved = net_input.detach().clone()
    net_input_kernel_saved = net_input_kernel.detach().clone()

    # store the loss values
    mse_list = []
    psnr_list = []
    ssim_gt_list = []
    psnr_gt_list = []
    losses_step = []

    # initialize variables for backtracking
    last_nets = None
    psnr_last = 0

    # initialize the variables for the MCMC
    sum_local_x_MCMC, sum_local_k_MCMC, num_local_MCMC = 0, 0, 0
    sum_global_x_MCMC, sum_global_k_MCMC, num_global_MCMC = 0, 0, 0

    ### start SelfDeblur
    for step in tqdm(range(num_iter),position=0, leave=True):

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

        # tuple with both the image and kernel networks
        nets = (net, net_kernel)

        # compute the loss
        if step < 1000:
            total_loss = mse(out_y, y)
        else:
            total_loss = 1-ssim(out_y, y)
        total_loss.backward()

        # SGLD optimization
        optimizer.step()

        # compute the psnr to the blurry image
        psnr = peak_signal_noise_ratio(y.detach().cpu().numpy()[0], out_y.detach().cpu().numpy()[0])

        if (step + 1) % opt.loss_frequency == 0:
            # compute the losses to the gt image
            out_x_cropped = out_x[:, :, padh // 2:padh // 2 + img_size[1], padw // 2:padw // 2 + img_size[2]]
            ssim_gt, psnr_gt = comparison_up_to_shift(x_gt.detach().cpu().numpy()[0, 0],
                                                      out_x_cropped.detach().cpu().numpy()[0, 0], maxshift=5)
            # store the losses to the gt image
            psnr_gt_list.append(psnr_gt)
            ssim_gt_list.append(ssim_gt)
            losses_step.append(step + 1)

        # store the losses to the blurry image
        mse_list.append(total_loss.item())
        psnr_list.append(psnr)

        # backtracking
        if roll_back and (step+1) % MCMC_iter:
            rolled_back, last_nets, psnr_last = backtracking(psnr,psnr_last,nets,last_nets)

        if (step+1) > burnin_iter:

            # if we are in the range to compute the average for the local MCMC_iteration
            if (step+1) % MCMC_iter <= averaging_iter//2 or ((step+1) + averaging_iter//2  % MCMC_iter) >= 0:
                sum_local_x_MCMC = out_x.detach().cpu() + sum_local_x_MCMC
                sum_local_k_MCMC = out_k_m.detach().cpu() + sum_local_k_MCMC
                num_local_MCMC += 1

                if (step+1) % MCMC_iter == averaging_iter//2:

                    # compute the average images
                    x_local_MCMC = sum_local_x_MCMC / num_local_MCMC
                    k_local_MCMC = sum_local_k_MCMC / num_local_MCMC

                    # save the images
                    save_path = os.path.join(opt.save_path,
                                             '%s_x_%d_it_local_MCMC.png' % (imgname, (step + 1) - averaging_iter //2 ))
                    save_image(x_local_MCMC, save_path, padh, padw, img_size)

                    save_path = os.path.join(opt.save_path,
                                             '%s_k_%d_it_local_MCMC.png' % (imgname, (step + 1) - averaging_iter // 2))
                    save_kernel(k_local_MCMC, save_path)

                    sum_local_x_MCMC, num_local_MCMC = 0, 0
                    sum_local_k_MCMC = 0

            if (step+1) % MCMC_iter == 0:
                sum_global_x_MCMC = out_x.detach().cpu() + sum_global_x_MCMC
                sum_global_k_MCMC = out_k_m.detach().cpu() + sum_global_k_MCMC
                num_global_MCMC += 1
                x_global_MCMC = sum_global_x_MCMC / num_global_MCMC
                k_global_MCMC = sum_global_k_MCMC / num_global_MCMC

                # save the current image
                save_path = os.path.join(opt.save_path, '%s_x_%d_it.png' % (imgname,(step+1)))
                save_image(out_x,save_path,padh,padw,img_size)

                # save the current kernel
                save_path = os.path.join(opt.save_path, '%s_k_%d_it.png' % (imgname, (step + 1)))
                save_kernel(out_k_m, save_path)

                # save the current MCMC global image
                save_path = os.path.join(opt.save_path, '%s_x_global_MCMC.png' % imgname)
                save_image(x_global_MCMC, save_path, padh, padw, img_size)

                # save the current MCMC global kernel
                save_path = os.path.join(opt.save_path, '%s_k_global_MCMC.png' % imgname)
                save_kernel(k_global_MCMC, save_path)

                # plot the psnr to the blurry image
                save_path = os.path.join(opt.save_path, '%s_MSE.png' % imgname)
                plt.figure()
                plt.plot(mse_list)
                plt.xlabel('Iterations',fontsize=15)
                plt.ylabel('MSE',fontsize=15)
                plt.savefig(save_path)
                plt.close()

                save_path = os.path.join(opt.save_path, '%s_PSNR.png' % imgname)
                plt.figure()
                plt.plot(psnr_list)
                plt.xlabel('Iterations', fontsize=15)
                plt.ylabel('PSNR', fontsize=15)
                plt.savefig(save_path)
                plt.close()

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



