import torch
import math


def add_noise_weights_model(models,param_noise_sigma,learning_rate,dtype):
    """ Adds white noise with std param_noise_sigma to the weights of the model for SGLD

    :param models: models for which noise is added to the weights
    :type models: tuple
    :param param_noise_sigma: standard deviation of the noise
    :type param_noise_sigma: int
    :param learning_rate: learning rate for training the model with SGLD
    :type learning_rate: float
    :param dtype: data type for the weights (torch.cuda.FloatTensor or torch.FloatTensor)
    :type dtype: torch.type
    :return: None
    """
    for model in models:
        for n in [x for x in model.parameters() if len(x.size()) == 4]:
            noise = torch.randn(n.size())*param_noise_sigma*learning_rate
            noise = noise.type(dtype)
            n.data = n.data + noise


def add_noise_gradients_model(models,param_noise_sigma,dtype):
    """ Adds white noise with std param_noise_sigma to the gradients for SGLD

    :param models: models for which noise is added to the weights
    :type models: tuple
    :param param_noise_sigma: standard deviation of the noise
    :type param_noise_sigma: int
    :param dtype: data type for the weights (torch.cuda.FloatTensor or torch.FloatTensor)
    :type dtype: torch.type
    :return: None
    """
    for model in models:
        for param in model.parameters():
            noise = torch.randn(param.shape) * math.sqrt(param_noise_sigma)
            noise = noise.type(dtype)
            param.grad += noise


def backtracking(psnr,psnr_last,nets,last_nets,threshold=5):
    """Backtracking in SGLD

    To solve the oscillation of model training, we go back to the previous model if the PSNR has
    largely dropped since the last backtracking check.

    :param psnr: current psnr value to the blurry image
    :type psnr: float
    :param float psnr_last: previous psnr value to the blurry image (from the last backtracking check)
    :type psnr_last: float
    :param nets: current networks (image net, kernel net)
    :type nets: tuple
    :param last_nets: parameters of the last networks (image net, kernel net)
    :type last_nets: tuple
    :param threshold: threshold to determine, defaults to -5
    :type threshold: int, optional
    :return: tuple (rolled_back, last_nets, last_psnr)
        WHERE
        bool rolled_back indicates if we rolled_back to the previous network
        tuple last_net is the parameters of the networks we stayed at or rolled back to
        float psnr is the psnr_last value obtained by the network we stayed at or rolled back to
    """

    # Backtracking
    rolled_back = False
    if psnr - psnr_last < -threshold and last_nets:
        print('Falling back to previous checkpoint.')
        for i in range(2):
            for new_param, net_param in zip(last_nets[i], nets[i].parameters()):
                net_param.detach().copy_(new_param.cuda())
        rolled_back = True

    else:
        last_nets = ([x.detach().cpu() for x in nets[0].parameters()],[x.detach().cpu() for x in nets[1].parameters()])
        psnr_last = psnr


    return rolled_back, last_nets, psnr_last

