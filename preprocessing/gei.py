import os
import cv2 as cv
import numpy as np


def mass_center(img, is_round=True):
    Y = img.mean(axis=1)
    X = img.mean(axis=0)
    Y_ = np.sum(np.arange(Y.shape[0]) * Y) / np.sum(Y)
    X_ = np.sum(np.arange(X.shape[0]) * X) / np.sum(X)
    if is_round:
        return int(round(X_)), int(round(Y_))
    return X_, Y_


def image_extract(img, newsize):
    x_s = np.where(img.mean(axis=0) != 0)[0].min()
    x_e = np.where(img.mean(axis=0) != 0)[0].max()

    y_s = np.where(img.mean(axis=1) != 0)[0].min()
    y_e = np.where(img.mean(axis=1) != 0)[0].max()

    x_c, _ = mass_center(img)
    x_s = x_c - newsize[1] // 2
    x_e = x_c + newsize[1] // 2
    img = img[
        y_s:y_e, x_s if x_s > 0 else 0: x_e if x_e < img.shape[1] else img.shape[1]
    ]
    return cv.resize(img, newsize)


def gei_():
    nor_silhouettes = os.listdir(os.path.join(os.getcwd(), 'normalize'))
    if len(nor_silhouettes) == 0:
        return False
    nor_silh = [
        cv.imread(os.path.join(os.getcwd(), 'normalize', frame), 0)
        for frame in nor_silhouettes
    ]
    nor_silh = [image_extract(frame, (128, 128)) for frame in nor_silh]
    gei = np.mean(nor_silh, axis=0).astype(np.uint8)
    cv.imwrite(
        f"{os.path.join(os.getcwd(),'gei')}/gei.png", gei)
    return True
