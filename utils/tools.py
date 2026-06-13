import time
import torch
import numpy as np
from math import exp
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import h5py

class ERGAS(torch.nn.Module):
    def __init__(self, ratio=4):
        super().__init__()
        self.ratio = ratio

    def forward(self, img, gt):
        b, c, _, _ = img.shape
        a1 = torch.mean((img - gt) ** 2, dim=(-2, -1))
        a2 = torch.mean(gt, dim=(-2, -1)) ** 2
        com = (a1 / a2).view(b, c)
        summ = torch.sum(com, dim=-1)
        ergas = 100 * (1 / self.ratio) * ((summ / c) ** 0.5)
        ergas = ergas.mean()
        return ergas


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channels=31):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channels
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):

        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def compute_ssim(img1, img2, window_size=7, size_average=True):
    (_, channel, _, _) = img1.size()


    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)


    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2


    max_val = img1.max()
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def SAM(pred, gt):

    pred = pred.flatten(1)
    gt = gt.flatten(1)
    dot = (pred * gt).sum(0)
    norm_pred = torch.norm(pred, p=2, dim=0)
    norm_gt = torch.norm(gt, p=2, dim=0)
    return torch.mean(torch.acos(dot / (norm_pred * norm_gt + 1e-8))) * 180 / np.pi

def compute_sam(pred, gt):

    if pred.dim() == 4:
        pred = pred.squeeze(0)
    if gt.dim() == 4:
        gt = gt.squeeze(0)


    pred_flat = pred.flatten(1)
    gt_flat = gt.flatten(1)


    dot = (pred_flat * gt_flat).sum(0)


    norm_pred = torch.norm(pred_flat, p=2, dim=0)
    norm_gt = torch.norm(gt_flat, p=2, dim=0)


    denominator = norm_pred * norm_gt
    mask = denominator < 1e-10
    denominator[mask] = 1.0


    cos_theta = dot / denominator
    cos_theta = torch.clamp(cos_theta, -1.0 + 1e-10, 1.0 - 1e-10)


    theta = torch.acos(cos_theta)


    theta[mask] = 0.0


    return torch.mean(theta) * 180 / np.pi

def RMSE(pred, gt):
    return torch.sqrt(torch.mean((pred - gt) ** 2))

def PSNR(pred, gt):


    pred = torch.clamp(pred, min=0.0, max=1.0)
    gt = torch.clamp(gt, min=0.0, max=1.0)

    mse = torch.mean((pred - gt)**2, dim=[2,3])


    mse = torch.clamp(mse, min=1e-8, max=10.0)


    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))


    psnr = torch.clamp(psnr, min=0.0, max=50.0)

    return torch.mean(psnr)

def compute_rmse(pred, gt):

    rmse_per_channel = torch.sqrt(torch.mean((pred - gt) ** 2, dim=(2, 3)))


    return torch.mean(rmse_per_channel)

def compute_psnr(pred, gt, max_val=1.0):

    pred = torch.clamp(pred, 0.0, max_val)
    gt = torch.clamp(gt, 0.0, max_val)


    mse = torch.mean((pred - gt) ** 2, dim=(2, 3))


    mse = torch.clamp(mse, min=1e-10)


    psnr = 10 * torch.log10((max_val ** 2) / mse)


    return torch.mean(psnr)

def save_matv73(file_path, var_name, data):
    with h5py.File(file_path, 'w') as f:
        f.create_dataset(var_name, data=data)


def gettime():
    current_time = time.localtime()
    time_str = str(current_time.tm_year) + '-' + str(current_time.tm_mon) + '-' + str(current_time.tm_mday) +\
               '-' + str(current_time.tm_hour)
    return time_str
