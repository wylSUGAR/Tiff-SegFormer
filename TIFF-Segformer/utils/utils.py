import random

import cv2
import numpy as np
import torch
from PIL import Image
from osgeo import gdal

#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#
def resize_image(image, size, tiff=None):
    iw, ih  = image.size
    w, h    = size

    scale   = min(w/iw, h/ih)
    nw      = int(iw*scale)
    nh      = int(ih*scale)

    image   = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))


    # tiff
    if tiff is not None:
        tiff = tiff.resize((nw, nh), Image.NEAREST)
        new_tiff = Image.new('L', [w, h], (0))
        new_tiff.paste(tiff, ((w - nw) // 2, (h - nh) // 2))

    return new_image, nw, nh, new_tiff
    
#---------------------------------------------------#
#   获得学习率
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

#---------------------------------------------------#
#   设置种子
#---------------------------------------------------#
def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#---------------------------------------------------#
#   设置Dataloader的种子
#---------------------------------------------------#
def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def preprocess_input(image):
    image -= np.array([123.675, 116.28, 103.53], np.float32)
    image /= np.array([58.395, 57.12, 57.375], np.float32)
    return image

def preprocess_tiff(tiff):
    tiff -= np.mean(np.array([123.675, 116.28, 103.53], np.float32))
    tiff /= np.mean(np.array([58.395, 57.12, 57.375], np.float32))
    return tiff

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

def download_weights(phi, model_dir="./model_data"):
    import os
    from torch.hub import load_state_dict_from_url
    
    download_urls = {
        'b0' : "https://github.com/bubbliiiing/segformer-pytorch/releases/download/v1.0/segformer_b0_backbone_weights.pth",
        'b1' : "https://github.com/bubbliiiing/segformer-pytorch/releases/download/v1.0/segformer_b1_backbone_weights.pth",
        'b2' : "https://github.com/bubbliiiing/segformer-pytorch/releases/download/v1.0/segformer_b2_backbone_weights.pth",
        'b3' : "https://github.com/bubbliiiing/segformer-pytorch/releases/download/v1.0/segformer_b3_backbone_weights.pth",
        'b4' : "https://github.com/bubbliiiing/segformer-pytorch/releases/download/v1.0/segformer_b4_backbone_weights.pth",
        'b5' : "https://github.com/bubbliiiing/segformer-pytorch/releases/download/v1.0/segformer_b5_backbone_weights.pth",
    }
    url = download_urls[phi]
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_state_dict_from_url(url, model_dir)


# -----------------------wyl
def tiff_to_array(tiff_path):
    tiff_raster = gdal.Open(tiff_path)
    tiff_band = tiff_raster.GetRasterBand(1)
    tiff_array = tiff_band.ReadAsArray()
    # return np.array(tiff_array) * 20
    # print("--------------------------------------tiff 转array" + ' '.join(str(x) for x in tiff_array))

    max_val, nor_tiff = min_max_normalize(tiff_array)
    # sobel_tiff = sobel_arr(nor_tiff * 20)
    # sobel_tiff = sobel_arr(tiff_array)
    # #
    # return np.multiply(np.array(sobel_tiff), np.array(tiff_array))
    # sn_arr = standard_normalize(tiff_array)
    # gaus_arr = standard_normalize(gaussian_filter(tiff_array, 6))
    return nor_tiff * 255


#  -----------------------wyl:返回最大值 和 最大最小值归一化
def min_max_normalize(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    normalized_arr = (arr - min_val) / (max_val - min_val)
    return max_val, normalized_arr


# -----------------------wyl:标准化
def standard_normalize(arr):
    return (arr - np.mean(arr)) / np.std(arr)


# ----------------------wyl: 计算sobel 算子
def sobel_arr(arr):
    arr_x_64 = cv2.Sobel(arr, cv2.CV_64F, dx=1, dy=0)
    arr_y_64 = cv2.Sobel(arr, cv2.CV_64F, dx=0, dy=1)
    arr_x_abs = cv2.convertScaleAbs(arr_x_64)
    arr_y_abs = cv2.convertScaleAbs(arr_y_64)
    dst = cv2.addWeighted(arr_x_abs, 0.5, arr_y_abs, 0.5, 0)
    return dst

# -----------------------wyl:高斯滤波
def gaussian_filter(matrix, sigma):
    # 计算高斯核
    kernel_size = np.ceil(2.5 * sigma).astype(int) + 1
    kernel_size = np.max((kernel_size, 3))  # 确保核至少是3x3
    grid_x, grid_y = np.mgrid[-kernel_size // 2 + 1:kernel_size // 2 + 1, -kernel_size // 2 + 1:kernel_size // 2 + 1]
    gaussian_kernel = np.exp(-(grid_x ** 2 + grid_y ** 2) / (2.0 * sigma ** 2))
    gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)

    # 对矩阵进行卷积
    import scipy.signal
    filtered_matrix = scipy.signal.convolve2d(matrix, gaussian_kernel, mode='same', boundary='fill', fillvalue=0)

    return filtered_matrix

