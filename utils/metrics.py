import numpy as np
import scipy.interpolate
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


def comparison_up_to_shift(im1,im2,maxshift):
    """ Comparing images up to a translation of maxshift and taking the best value

    Need to account for shift because the kernel reconstruction is shift invariant - a small
    shift of the image and kernel will not effect the likelihood score.

    Translated from matlab code from https://github.com/csdwren/SelfDeblur/blob/master/statistic/comp_upto_shift.m

    :param im1: image 1
    :type im1: np.array
    :param im1: image 2
    :type im2: np.array
    :param maxshift: maximum shift for the comparison
    :type maxshift: int
    :return: tuple (ssim, psn)
        WHERE
        float ssim is the SSIM value corresponding the best value within the maxshift translation range
        float psnr is the PSNR value corresponding the best value within the maxshift translation range
    """

    step = 0.25
    shifts = np.arange(-maxshift,maxshift+1,step)

    im2 = im2[15:-15, 15:-15]
    im1 = im1[15-maxshift:-15+maxshift+1, 15-maxshift:-15+maxshift+1]
    n1, n2 = im2.shape[0], im2.shape[1]
    (gx,gy) = np.meshgrid(np.arange(-maxshift,n2+maxshift+1),np.arange(-maxshift,n1+maxshift+1))
    (gx0, gy0) = np.meshgrid(np.arange(0,n2), np.arange(0,n1))

    ssdem = np.zeros((len(shifts),len(shifts)))

    for i in range(len(shifts)):
        for j in range(len(shifts)):
            xn = np.arange(0,n2) + shifts[i]
            yn = np.arange(0,n1) + shifts[j]

            f = scipy.interpolate.RectBivariateSpline(np.arange(-maxshift,n2+maxshift+1),np.arange(-maxshift,n1+maxshift+1),im1)
            tim1 = f(xn,yn,grid=True)

            ssdem[i,j] = sum(sum((tim1 - im2)**2))

    k = np.argmin(ssdem)
    i, j = k // ssdem.shape[1], k % ssdem.shape[1]

    xn = np.arange(0, n2) + shifts[i]
    yn = np.arange(0, n1) + shifts[j]
    f = scipy.interpolate.RectBivariateSpline(np.arange(-maxshift,n2+maxshift+1),np.arange(-maxshift,n1+maxshift+1), im1)
    tim1 = f(xn,yn)

    psnr = peak_signal_noise_ratio(tim1,im2)
    ssim = structural_similarity(tim1, im2)

    return ssim, psnr





