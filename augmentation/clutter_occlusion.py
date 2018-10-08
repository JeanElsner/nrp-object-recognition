import cv2
import glob
import numpy as np
import random
import os

# TODO make command line script
clutter_percentage = .9
path = r'C:\Users\jeane\Desktop\GitHub\Matrix-Capsules-EM-PyTorch\data\Dataset_lighting4\left\*.png'
files = glob.glob(path)


def rotated_triangle(y, x, baseline, height, theta):
    hh = height / 2.
    hb = baseline / 2.
    c, s = (np.cos(theta), np.sin(theta))
    R = np.array(((c, -s), (s, c)))
    p1 = np.matmul((0, - hh), R) + (x, y)
    p2 = np.matmul((- hb, + hh), R) + (x, y)
    p3 = np.matmul((+ hb, + hh), R) + (x, y)
    return np.array((p1, p2, p3), dtype=int)


def rotated_rectangle(y, x, width, height, theta):
    hh = height / 2.
    hw = width / 2.
    c, s = (np.cos(theta), np.sin(theta))
    R = np.array(((c, -s), (s, c)))
    p1 = np.matmul((- hw, - hh), R) + (x, y)
    p2 = np.matmul((- hw, + hh), R) + (x, y)
    p3 = np.matmul((+ hw, + hh), R) + (x, y)
    p4 = np.matmul((+ hw, - hh), R) + (x, y)
    return np.array((p1, p2, p3, p4), dtype=int)


def add_clutter(im, percentage, width=(7, 28), height=(7, 14), color=(1, 255), angle=(0, 180)):
    non_zero = percentage * im.size

    while (cv2.countNonZero(im) < non_zero):
        ry = random.randint(0, im.shape[0] - 1)
        rx = random.randint(0, im.shape[1] - 1)
        rheight = random.randint(height[0], height[1])
        rwidth = random.randint(width[0], width[1])
        rcolor = random.randint(color[0], color[1])
        rangle = random.randint(angle[0], angle[1])
        case = random.randint(0, 2)
        if case == 0:
            pts = rotated_rectangle(ry, rx, rwidth, rheight, rangle)
            cv2.fillConvexPoly(im, pts, rcolor)
        elif case == 1:
            axis1 = int(rwidth / 2)
            axis2 = int(rheight / 2)
            cv2.ellipse(im, (rx, ry), (axis1, axis2), rangle, 0, 360, rcolor, -1)
        elif case == 2:
            pts = rotated_triangle(rx, ry, rwidth, rheight, rangle)
            cv2.fillConvexPoly(im, pts, rcolor)


for f in files:
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    clutter = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    add_clutter(clutter, clutter_percentage)
    __, mask = cv2.threshold(clutter, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask, mask)
    im_clutter = cv2.add(cv2.bitwise_and(img, mask_inv), clutter)
    path_dir, filename = os.path.split(f)
    directory = os.path.join(path_dir, 'clutter_{}'.format(clutter_percentage))
    if not os.path.exists(directory):
        os.makedirs(directory)
    cv2.imwrite(os.path.join(directory, filename), im_clutter)
