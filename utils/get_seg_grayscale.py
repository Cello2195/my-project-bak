import numpy as np
from PIL import Image
import cv2


def get_seg_grayscale():
    img = Image.open('tmp/resized_segmentation_img.png')
    img_w, img_h = img.size

    img = np.array(img)
    gray_img = np.zeros((img_h, img_w))

    for y_idx in range(img.shape[0]):
        for x_idx in range(img.shape[1]):
            tmp = img[y_idx][x_idx]
            if np.array_equal(tmp, [0, 0, 0]):
                gray_img[y_idx][x_idx] = 0
            if np.array_equal(tmp, [255, 0, 0]):
                gray_img[y_idx][x_idx] = 2
            elif np.array_equal(tmp, [0, 0, 255]):
                gray_img[y_idx][x_idx] = 13
            elif np.array_equal(tmp, [85, 51, 0]):
                gray_img[y_idx][x_idx] = 10
            elif np.array_equal(tmp, [255, 85, 0]):
                gray_img[y_idx][x_idx] = 5
            elif np.array_equal(tmp, [0, 255, 255]):
                gray_img[y_idx][x_idx] = 15
            elif np.array_equal(tmp, [51, 170, 221]):
                gray_img[y_idx][x_idx] = 14
            elif np.array_equal(tmp, [0, 85, 85]):
                gray_img[y_idx][x_idx] = 9
            elif np.array_equal(tmp, [0, 0, 85]):
                gray_img[y_idx][x_idx] = 6
            elif np.array_equal(tmp, [0, 128, 0]):
                gray_img[y_idx][x_idx] = 12
            elif np.array_equal(tmp, [177, 255, 85]):
                gray_img[y_idx][x_idx] = 17
            elif np.array_equal(tmp, [85, 255, 170]):
                gray_img[y_idx][x_idx] = 16
            elif np.array_equal(tmp, [0, 119, 221]):
                gray_img[y_idx][x_idx] = 5
            else:
                gray_img[y_idx][x_idx] = 0

    img = cv2.resize(gray_img, (768, 1024), interpolation=cv2.INTER_NEAREST)
    bg_img = Image.fromarray(np.uint8(img), "L")
    bg_img.save("./repositories/HR-VITON-main/test/test/image-parse-v3/00001_00.png")


if __name__ == '__main__':
    get_seg_grayscale()