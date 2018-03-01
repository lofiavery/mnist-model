# https://medium.com/@o.kroeger/tensorflow-mnist-and-your-own-handwritten-digits-4d1cd32bbab4
import numpy as np
import glob
import os
import cv2
import math
from scipy import ndimage


class ImageHandler(object):
    imgs_dir = "D:\\MLIternshipMihlala\\exe1\\test_mnist\\"
    processed = "D:\\MLIternshipMihlala\\exe1\\processed_mnist\\"
    def _init_(self):
        self.num_of_classes = 10
    def getBestShift(self, img):
        cy, cx = ndimage.measurements.center_of_mass(img)
        rows, cols = img.shape
        shiftx = np.round(cols / 2.0 - cx).astype(int)
        shifty = np.round(rows / 2.0 - cy).astype(int)
        return shiftx, shifty

    def shift(self, img, sx, sy):
        rows, cols = img.shape
        M = np.float32([[1, 0, sx], [0, 1, sy]])
        shifted = cv2.warpAffine(img, M, (cols, rows))
        return shifted

    def transform_to_20X20(self, gray):
        while np.sum(gray[0]) == 0:
            gray = gray[1:]
        while np.sum(gray[:, 0]) == 0:
            gray = np.delete(gray, 0, 1)
        while np.sum(gray[-1]) == 0:
            gray = gray[:-1]
        while np.sum(gray[:, -1]) == 0:
            gray = np.delete(gray, -1, 1)
        rows, cols = gray.shape

        if rows > cols:
            factor = 20.0 / rows
            rows = 20
            cols = int(round(cols * factor))
            gray = cv2.resize(gray, (cols, rows))
        else:
            factor = 20.0 / cols
            cols = 20
            rows = int(round(rows * factor))
            gray = cv2.resize(gray, (cols, rows))

        colsPadding = (int(math.ceil((28 - cols) / 2.0)), int(math.floor((28 - cols) / 2.0)))
        rowsPadding = (int(math.ceil((28 - rows) / 2.0)), int(math.floor((28 - rows) / 2.0)))
        gray = np.lib.pad(gray, (rowsPadding, colsPadding), 'constant')
        shiftx, shifty = self.getBestShift(gray)
        shifted = self.shift(gray, shiftx, shifty)
        gray = shifted
        return gray

    def parse_images(self,src=imgs_dir,dest=processed):
        self.imgs_dir = src
        self.processed = dest
        size = len(os.listdir(self.imgs_dir))
        # create an array where we can store our pictures
        images = np.zeros((size, 784))
        # and the correct values
        correct_vals = np.zeros((size, 10))
        i = 0
        for im_name in os.listdir(self.imgs_dir):
            # read the image
            gray = cv2.imread(self.imgs_dir + im_name, cv2.IMREAD_GRAYSCALE)
            # resize the images and invert it (black background)
            gray = cv2.resize(255 - gray, (28, 28))
            # invert the gray background to black everything above 128 => 255
            (thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            # 28X28 => 20X20 => 28X28
            gray = self.transform_to_20X20(gray)
            # save the processed images
            cv2.imwrite(self.processed + im_name, gray)
            """
            all images in the training set have an range from 0-1
            and not from 0-255 so we divide our flatten images
            (a one dimensional vector with our 784 pixels)
            to use the same 0-1 based range
            """
            flatten = gray.flatten() / 255.0
            """
            we need to store the flatten image and generate
            the correct_vals array
            correct_val for the first digit (9) would be
            [0,0,0,0,0,0,0,0,0,1]
            """
            images[i] = flatten
            correct_val = np.zeros((10))
            correct_val[int(im_name[:1])] = 1
            correct_vals[i] = correct_val
            i += 1
        return (images, correct_vals)