import numpy as np
import av
import matplotlib.pyplot as plt
import cv2
from skimage.color import rgb2gray
from skimage.feature import peak_local_max
from scipy.signal import convolve2d
from PIL import Image, ImageDraw
from sklearn.linear_model import LinearRegression
import os

def convert_to_pil(arr):
    arr = arr * 256
    return Image.fromarray(arr).convert('RGB')

def save_image(arr, path):
    convert_to_pil(arr).save(path)

def get_number_image(frame, finish_line, horz_comp, file_name):
    for i in range(len(horz_comp)):
        if horz_comp[i] == 0:
            frame[i,:] = np.zeros(frame[i,:].shape)
    save_image(frame, file_name)

base = np.load('base_image.npy')
im = np.load('image300.npy')
edges = np.load('edges.npy')

sub = abs(im - base)
#save_image(sub, 'visuals/frame300_subtracted.png')

vert_filter = np.ones((1, im.shape[1])) / im.shape[1]
horz_filtered = convolve2d(sub, vert_filter, mode='valid')
#save_image(horz_filtered, 'visuals/horizontal_filtered.png')
horz_bg = horz_filtered < .03
horz_person = horz_filtered > .03
horz_filtered[horz_bg] = 0
horz_filtered[horz_person] = 1
#save_image(horz_filtered, 'visuals/horizontal_filtered_norm.png')
for i in range(740, 750):
    slice = edges[i, :].reshape(1, len(edges[i,:]))
    horz_bg = slice < .2
    horz_person = slice > .2
    slice[horz_bg] = 0
    slice[horz_person] = 1
    save_image(slice, 'visuals/slice_' + str(i) + '.png')