import numpy as np
import cv2
import math


def bilinear_interpolation(src, dst_size):
    src_h, src_w, channel = src.shape
    dst_h, dst_w = dst_size[0], dst_size[1]
    if src_h == dst_h and src_w == dst_w:
        return src.copy()

    dst = np.zeros((dst_h, dst_w, channel), dtype=np.uint8)
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_w
    for current_channel in range(channel):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                x = (dst_x + 0.5) * scale_x - 0.5
                y = (dst_y + 0.5) * scale_y - 0.5

                x0 = int(np.floor(x))
                x1 = min(x0 + 1, src_w - 1)  # 不要超过边界
                y0 = int(np.floor(y))
                y1 = min(y0 + 1, src_h - 1)

                temp0 = src[y0, x0, current_channel] * (x1 - x) + src[y0, x1, current_channel] * (x - x0)
                temp1 = src[y1, x0, current_channel] * (x1 - x) + src[y1, x1, current_channel] * (x - x0)

                res = (y1 - y) * temp0 + (y - y0) * temp1
                dst[dst_y, dst_x, current_channel] = int(res)

    return dst


if __name__ == '__main__':
    img = cv2.imread('timg.jpeg')
    dst = bilinear_interpolation(img, (255, 255))
    cv2.imshow('result', dst)
    cv2.waitKey(0)
