import numpy as np
import math
import glob
import cv2
import matplotlib.pyplot as plt
import os

# TODO make command line script
path = r'C:\Users\jeane\Desktop\GitHub\Matrix-Capsules-EM-PyTorch\data\Dataset_lighting4\left\*.png'
files = glob.glob(path)
tile_num = 3


def imshow(im):
    plt.imshow(im)


def get_tile_shape(im, tile_num):
    tile_size_x = int(im.shape[0]/tile_num)
    tile_size_y = int(im.shape[1]/tile_num)
    return tile_size_x, tile_size_y


def get_tile(im, tile, tile_num):
    shape = get_tile_shape(im, tile_num)
    idx = int(math.floor(tile/tile_num))
    idy = tile%tile_num
    return im[idx*shape[0]:(idx+1)*shape[0],idy*shape[1]:(idy+1)*shape[1]]


def set_tile(im, tile, tile_im):
    shape = get_tile_shape(im, tile_num)
    idx = int(math.floor(tile/tile_num))
    idy = tile%tile_num
    im[idx*shape[0]:(idx+1)*shape[0],idy*shape[1]:(idy+1)*shape[1]] = tile_im


def permute_tiles(im, tile_num):
    im_perm = np.zeros(im.shape, dtype=np.uint8)
    perm = np.random.permutation(tile_num**2)
    for idx in range(0, tile_num):
        for idy in range(0, tile_num):
            set_tile(im_perm, perm[idx*tile_num+idy], get_tile(im, idx*tile_num+idy, tile_num))
    return im_perm


for f in files:
    im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    tiles = permute_tiles(im, tile_num)
    path_dir, filename = os.path.split(f)
    directory = os.path.join(path_dir, f'tiles_{tile_num}')
    if not os.path.exists(directory):
        os.makedirs(directory)
    cv2.imwrite(os.path.join(directory, filename), tiles)
