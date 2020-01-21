from pathlib import Path

import numpy as np
import cv2
from matplotlib import pyplot as plt


class RemoveBorders(object):
    @classmethod
    def __remove_borders(cls, img, threshold=None, otsu=False):
        if threshold is None and otsu is False:
            raise ValueError("bad parameters")

        grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grey_img, (5, 5), 0)
        ret3, th3 = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if otsu:
            threshold = ret3

        img_thresh = grey_img > threshold
        mask = np.argwhere(img_thresh)

        y_min = np.min(mask[:, 0])
        y_max = np.max(mask[:, 0])
        x_min = np.min(mask[:, 1])
        x_max = np.max(mask[:, 1])
        img = img[y_min:y_max, x_min:x_max]
        return img

    @staticmethod
    def otsu(img):
        return RemoveBorders.__remove_borders(img, otsu=True)

    @staticmethod
    def binary(img, threshold=200):
        return RemoveBorders.__remove_borders(img, threshold, otsu=False)


if __name__ == '__main__':
    for img_fp in Path("data").iterdir():
        img = cv2.imread(str(img_fp))
        plt.subplot(121)
        plt.title("Otsu")
        out_img = RemoveBorders.otsu(img)
        plt.imshow(out_img)
        plt.subplot(122)
        plt.title("Binary")
        out_img = RemoveBorders.binary(img)
        plt.imshow(out_img)
        plt.show()
