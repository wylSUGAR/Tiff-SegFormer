import numpy as np
import math
import PIL
import cv2
import os


def rgb2hsv(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx - mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g - b) / df) + 360) % 360
    elif mx == g:
        h = (60 * ((b - r) / df) + 120) % 360
    elif mx == b:
        h = (60 * ((r - g) / df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = df / mx
    v = mx
    return h, s, v


def hsv2rgb(h, s, v):
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0:
        r, g, b = v, t, p
    elif hi == 1:
        r, g, b = q, v, p
    elif hi == 2:
        r, g, b = p, v, t
    elif hi == 3:
        r, g, b = p, q, v
    elif hi == 4:
        r, g, b = t, p, v
    elif hi == 5:
        r, g, b = v, p, q
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return r, g, b


# #--------------wyl:8位调色码生成
def label_colormap(n_label=256, value=None):
    """Label colormap.

    Parameters
    ----------
    n_labels: int
        Number of labels (default: 256).
    value: float or int
        Value scale or value of label color in HSV space.

    Returns
    -------
    cmap: numpy.ndarray, (N, 3), numpy.uint8
        Label id to colormap.

    """

    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    cmap = np.zeros((n_label, 3), dtype=np.uint8)
    for i in range(0, n_label):
        id = i
        r, g, b = 0, 0, 0
        for j in range(0, 8):
            r = np.bitwise_or(r, (bitget(id, 0) << 7 - j))
            g = np.bitwise_or(g, (bitget(id, 1) << 7 - j))
            b = np.bitwise_or(b, (bitget(id, 2) << 7 - j))
            id = id >> 3
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b

    if value is not None:
        hsv = rgb2hsv(cmap.reshape(1, -1, 3))
        if isinstance(value, float):
            hsv[:, 1:, 2] = hsv[:, 1:, 2].astype(float) * value
        else:
            assert isinstance(value, int)
            hsv[:, 1:, 2] = value
        cmap = hsv2rgb(hsv).reshape(-1, 3)
    return cmap


#--------------wyl:将label转换为RGB图片
def lblsave(filename, lbl):
    if filename[-4:] != '.png':
        filename += '.png'
    # Assume label ranses [-1, 254] for int32,
    # and [0, 255] for uint8 as VOC.
    if lbl.min() >= -1 and lbl.max() < 255:
        lbl_pil = PIL.Image.fromarray(lbl.astype(np.uint8), mode='P')
        colormap = label_colormap()
        lbl_pil.putpalette(colormap.flatten())
        lbl_pil.save(filename)
    else:
        raise ValueError(
            '[%s] Cannot save the pixel-wise class label as PNG. '
            'Please consider using the .npy format.' % filename
        )


#------------------wyl:24位深图转换为8位深图
def covert_24_8(dir_24, out_8_dir):
    # 读取调色板
    colormap = label_colormap()

    # 读入图片并将opencv的BGR转换为RGB格式
    for img in os.listdir(dir_24):
        if img[-4:] == '.png':
            img_basename = os.path.basename(img)
            img = cv2.imread(os.path.join(dir_24, img))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 将24位深图片中的[r,g,b]对应到colormap反求出label
            lbls = np.zeros(shape=[img.shape[0], img.shape[1]], dtype=np.int32)
            len_colormap = len(colormap)
            indexes = np.nonzero(img)

            for i, j in zip(indexes[0], indexes[1]):
                for k in range(len_colormap):
                    if all(img[i, j, :3] == colormap[k]):
                        lbls[i, j] = k
                        break

            #####
            ##### 此处添加对lal图的变换处理过程，注意数据类型为int32
            #####

            # 将label再转换成8位label.png
            lblsave(os.path.join(out_8_dir, os.path.basename(img_basename)), lbls)


if __name__ == '__main__':
    dir_24 = "img_out"
    out_8_dir = "img_out/8"
    covert_24_8(dir_24, out_8_dir)






