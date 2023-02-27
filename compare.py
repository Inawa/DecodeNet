from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse

import lpips


def calc_ssim(img1_path, img2_path,num):

    ssim_score = 0
    for i in range(num):
        path1 = img1_path + "rel_" + str(i) + ".png"
        path2 = img2_path + "res_" + str(i) + ".png"
        img1 = Image.open(path1).convert('L')
        img2 = Image.open(path2).convert('L')
        img1, img2 = np.array(img1), np.array(img2)
        ssim_score += ssim(img1, img2, data_range=255)
    return ssim_score/num


def calc_psnr(img1_path, img2_path,num):
    psnr_score = 0
    for i in range(num):
        path1 = img1_path + "rel_" + str(i) + ".png"
        path2 = img2_path + "res_" + str(i) + ".png"
        img1 = Image.open(path1)
        img2 = Image.open(path2)
        img1, img2 = np.array(img1)/255, np.array(img2)/255
        psnr_score += psnr(img1, img2, data_range=1)
    return psnr_score/num

def calc_mse(img1_path, img2_path,num):
    mse_score = 0
    for i in range(num):
        path1 = img1_path + "rel_" + str(i) + ".png"
        path2 = img2_path + "res_" + str(i) + ".png"
        img1 = Image.open(path1)
        img2 = Image.open(path2)
        img1, img2 = np.array(img1)/255, np.array(img2)/255

        mse_score += mse(img1, img2)
    return mse_score/num


loss_fn = lpips.LPIPS(net='alex')
def calc_lpips(img1_path, img2_path,num):
    dist01 = 0
    for i in range(num):
        path1 = img1_path + "rel_" + str(i) + ".png"
        path2 = img2_path + "res_" + str(i) + ".png"
        # Load images
        img0 = lpips.im2tensor(lpips.load_image(path1))  # RGB image from [-1,1]
        img1 = lpips.im2tensor(lpips.load_image(path2))
        dist01 += loss_fn.forward(img0, img1)
    return dist01/num



class util_of_lpips():
    def __init__(self, net, use_gpu=False):

        self.use_gpu = use_gpu
        if use_gpu:
            self.loss_fn.cuda()

    

