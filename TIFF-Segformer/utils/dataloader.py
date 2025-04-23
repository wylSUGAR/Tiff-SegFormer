import os

import cv2
import numpy as np
import torch
from PIL import Image
from PIL import ImageEnhance
from torch.utils.data.dataset import Dataset
from utils.utils import preprocess_input, cvtColor, tiff_to_array, standard_normalize, preprocess_tiff


class SegmentationDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
        super(SegmentationDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.length             = len(annotation_lines)
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.train              = train
        self.dataset_path       = dataset_path

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name            = annotation_line.split()[0]

        #-------------------------------#
        #   从文件中读取图像
        #-------------------------------#
        jpg         = Image.open(os.path.join(os.path.join(self.dataset_path, "VOC2007/JPEGImages"), name + ".jpg"))
        png         = Image.open(os.path.join(os.path.join(self.dataset_path, "VOC2007/SegmentationClass"), name + ".png"))

        tiff_path = os.path.join(os.path.join(self.dataset_path, "VOC2007/TiffImages"), name + ".tiff")
        tiff = tiff_to_array(tiff_path)

        #-------------------------------#
        #   数据增强
        #-------------------------------#
        jpg, png, tiff    = self.get_random_data(jpg, png, tiff, self.input_shape, random = self.train)


        jpg         = np.transpose(preprocess_input(np.array(jpg, np.float64)), [2,0,1])
        png         = np.array(png)

        # tiff:  先归一化
        tiff = preprocess_tiff(np.array(tiff, np.float32))
        #tiff: 增加tiff维度，由(512,512)   ---> (1,512,512)
        tiff        = np.expand_dims(np.array(tiff, np.float32), axis=0)


        png[png >= self.num_classes] = self.num_classes
        #-------------------------------------------------------#
        #   转化成one_hot的形式
        #   在这里需要+1是因为voc数据集有些标签具有白边部分
        #   我们需要将白边部分进行忽略，+1的目的是方便忽略。
        #-------------------------------------------------------#
        seg_labels  = np.eye(self.num_classes + 1)[png.reshape([-1])]
        seg_labels  = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

        return jpg, png, seg_labels, tiff

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, tiff, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        image   = cvtColor(image)
        # print("----------------------------------------------img"+' '.join(str(x) for x in np.array(image)))
        label   = Image.fromarray(np.array(label))

        # ------------wyl
        tiff    = Image.fromarray(np.array(tiff, np.float32))
        # ------------------------wyl

        # -------------wyl :图像增强：对比度增强、亮度增强、颜色增强、高斯噪声、运动模糊
        # image = enhance_img(image)
        # -------------wyl

        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = image.size
        h, w    = input_shape

        if not random:
            iw, ih  = image.size
            scale   = min(w/iw, h/ih)
            nw      = int(iw*scale)
            nh      = int(ih*scale)

            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', [w, h], (128, 128, 128))
            new_image.paste(image, ((w-nw)//2, (h-nh)//2))

            label       = label.resize((nw,nh), Image.NEAREST)
            new_label   = Image.new('L', [w, h], (0))
            new_label.paste(label, ((w-nw)//2, (h-nh)//2))

            #tiff
            tiff = tiff.resize((nw, nh), Image.NEAREST)
            new_tiff = Image.new('L', [w, h], (0))
            new_tiff.paste(tiff, ((w - nw) // 2, (h - nh) // 2))


            return new_image, new_label, new_tiff

        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(0.5, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)
        label = label.resize((nw,nh), Image.NEAREST)
        # -----------------wyl:tiff
        tiff = tiff.resize((nw,nh), Image.NEAREST)
        
        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: 
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
            #tiff
            tiff = tiff.transpose(Image.FLIP_LEFT_RIGHT)

        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_label = Image.new('L', (w,h), (0))
        #tiff
        new_tiff = Image.new('L', (w, h), (0))

        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        #tiff
        new_tiff.paste(tiff, (dx, dy))

        image = new_image
        label = new_label
        #tiff
        tiff = new_tiff

        image_data      = np.array(image, np.uint8)
        #------------------------------------------#
        #   高斯模糊
        #------------------------------------------#
        blur = self.rand() < 0.35
        if blur: 
            image_data = cv2.GaussianBlur(image_data, (5, 5), 0)

        #------------------------------------------#
        #   旋转
        #------------------------------------------#
        rotate = self.rand() < 0.25
        if rotate: 
            center      = (w // 2, h // 2)
            rotation    = np.random.randint(-10, 11)
            M           = cv2.getRotationMatrix2D(center, -rotation, scale=1)
            image_data  = cv2.warpAffine(image_data, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(128,128,128))
            label       = cv2.warpAffine(np.array(label, np.uint8), M, (w, h), flags=cv2.INTER_NEAREST, borderValue=(0))
            #tiff
            tiff       = cv2.warpAffine(np.array(tiff, np.uint8), M, (w, h), flags=cv2.INTER_NEAREST, borderValue=(0))

        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        
        return image_data, label, tiff


def seg_dataset_collate(batch):
    images      = []
    pngs        = []
    seg_labels  = []
    # ------wyl
    tiffs        = []
    for img, png, labels, tiff in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
        tiffs.append(tiff)
    images      = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs        = torch.from_numpy(np.array(pngs)).long()
    seg_labels  = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    tiffs       = torch.from_numpy(np.array(tiffs)).type(torch.FloatTensor)
    return images, pngs, seg_labels, tiffs


'''
函 数 名：contrastEnhancement(root_path, img_name, contrast)
函数功能：对比度增强
入口参数：
        root_path ：图片根目录
        img_name ：图片名称
        contrast ：对比度
返 回 值：
        对比度增强后的图片
'''
def contrastEnhancement(image, contrast):
    # image = Image.open(os.path.join(root_path, img_name))
    enh_con = ImageEnhance.Contrast(image)
    image_contrasted = enh_con.enhance(contrast)
    return image_contrasted


'''
函 数 名：brightnessEnhancement(root_path,img_name,brightness)
函数功能：亮度增强
入口参数：
        root_path ：图片根目录
        img_name ：图片名称
        brightness ：亮度
返 回 值：
        亮度增强后的图片
'''
def brightnessEnhancement(image, brightness):
    # image = Image.open(os.path.join(root_path, img_name))
    enh_bri = ImageEnhance.Brightness(image)
    image_brightened = enh_bri.enhance(brightness)
    return image_brightened


'''
函 数 名：colorEnhancement(root_path,img_name,color)
函数功能：颜色增强
入口参数：
        root_path ：图片根目录
        img_name ：图片名称
        color ：颜色
返 回 值：
        颜色增强后的图片
'''
def colorEnhancement(image,color):
    # image = Image.open(os.path.join(root_path, img_name))
    enh_col = ImageEnhance.Color(image)
    image_colored = enh_col.enhance(color)
    return image_colored


'''
函 数 名：gaussian_noise(img, mean, sigma)
函数功能：添加高斯噪声
入口参数：
        img ：原图
        mean ：均值
        sigma ：标准差
返 回 值：
        噪声处理后的图片
'''
def gaussian_noise(img, mean=0.1, sigma=0.08):
    img = np.array(img)
    # 将图片灰度标准化
    img = (img / 255)
    # 产生高斯 noise
    noise = np.random.normal(mean, sigma, img.shape)
    # 将噪声和图片叠加
    gaussian_out = img + noise
    # 将超过 1 的置 1，低于 0 的置 0
    gaussian_out = np.clip(gaussian_out, 0, 1)
    # 将图片灰度范围的恢复为 0-255
    gaussian_out = np.uint8(gaussian_out*255)
    # 这里也会返回噪声，注意返回值
    return Image.fromarray(gaussian_out)




'''
函 数 名：motion_blur(image)
函数功能：运动模糊化处理
入口参数：
        image ：原图
返 回 值：
        模糊化处理后的图片
'''
def motion_blur(image, degree=50, angle=45):
    image = np.array(image)

    # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)

    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return Image.fromarray(blurred)



def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a

def randIn(a=40, b=60):
    return np.random.randint(a, b)

'''
函 数 名：enhance_img(input_dir, output_dir)
函数功能：将原图进行高斯和模糊化变换
入口参数：
        input_dir ：原图路径
        output_dir ：变换后路径
返 回 值：
        无
'''
def enhance_img(image):
    # 判断对比度增强、亮度增强还是颜色增强
    randin = randIn(0, 5)

    # 对比度增强
    if randin == 0:
        image = contrastEnhancement(image=image, contrast=rand(0.6, 1.5))
    # 亮度增强
    if randin == 1:
        image = brightnessEnhancement(image=image, brightness=rand(0.6, 1.5))
    # 颜色增强
    if randin == 2:
        image = colorEnhancement(image=image, color=rand(0.6, 1.5))
    # 运动模糊
    if randin == 3:
        image = motion_blur(image=image, degree=randIn(45, 50), angle=randIn(40, 50))
    # 高斯噪声
    if randin == 4:
        image = gaussian_noise(img=image, mean=rand(0, 0.5), sigma=rand(0, 0.4))
    return image