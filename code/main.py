import numpy as np
import av
import matplotlib.pyplot as plt
import cv2
from skimage.color import rgb2gray
from skimage.feature import peak_local_max
from scipy.signal import convolve2d
from PIL import Image, ImageDraw
from sklearn.linear_model import LinearRegression
from scipy.ndimage.filters import gaussian_filter1d
import os

def convert_to_pil(arr):
    arr = arr * 256
    return Image.fromarray(arr).convert('RGB')

def save_image(arr, path):
    convert_to_pil(arr).save(path)

def get_number_image(frame, base_image, finish_line, horz_comp, file_name):
    horz_comp2 = gaussian_filter1d(horz_comp, sigma=5)
    lower_bound = finish_line - 20
    for i in range(finish_line - 20, 0, -1):
        if horz_comp2[i] < .1:
            lower_bound = i
            break
    sub = abs(frame - base_image)
    segment = np.transpose(frame[lower_bound:finish_line+10, :])
    segment_filter = np.transpose(sub[lower_bound:finish_line+10, :])
    horz_filter = np.ones((1, segment.shape[1])) / segment.shape[1]
    
    horz_filtered = convolve2d(segment_filter, horz_filter, mode='valid').reshape(segment.shape[0])
    horz_filtered_g = horz_filtered > 0.05
    horz_filtered_l = horz_filtered <= 0.05
    horz_filtered[horz_filtered_g] = 1
    horz_filtered[horz_filtered_l] = 0
    segment_height = int(np.sum(horz_filtered))
    output = np.zeros((segment_height, segment.shape[1]))
    cntr = 0
    for i in range(len(horz_filtered)):
        if horz_filtered[i] == 1:
            output[cntr,:] = segment[i,:]
            cntr += 1
    save_image(output, file_name)

finish_line = 360
frames = 0
horizontal_comp = 0
edges = 0
finish_image = 0

if True or not os.path.isfile('edges.npy') or not os.path.isfile('finish.npy') or not os.path.isfile('horizontal_comp.npy') or not os.path.isfile('frames.npy'):
    path = '../data/finish-line/20190413_142512.mp4'
    video = cv2.VideoCapture(path)
    container = av.open(path)

    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    out_image = 0


    base_image = 0


    vert_filter = 0
    horz_filter = 0
    print('loaded video')
    i = 0
    
    
    vertical_comp = 0

    for frame in container.decode(video=0):
        image = rgb2gray(np.array(frame.to_image()))
        if i == 0:
            out_image = np.zeros((num_frames, image.shape[0]))
            finish_image = np.zeros((image.shape[1], num_frames))
            base_image = image
            vert_filter = np.ones((1, image.shape[1])) / image.shape[1]
            horz_filter = np.ones((image.shape[0], 1)) / image.shape[0]
            frames = np.zeros((num_frames, image.shape[0], image.shape[1]))
            horizontal_comp = np.zeros((num_frames, image.shape[0]))
            vertical_comp = np.zeros((num_frames, image.shape[1]))
        finish_image[:, i] = image[finish_line, :]
        if i == 300:
            np.save('image300.npy', image)
            np.save('base_image.npy', base_image)
        frames[i,:,:] = image
        image = abs(image - base_image)
        
        horz_filtered = convolve2d(image, vert_filter, mode='valid')
        horz_bg = horz_filtered < .04
        horz_person = horz_filtered > .04
        horz_filtered[horz_bg] = 0
        horz_filtered[horz_person] = 1

        horizontal_comp[i, :] = horz_filtered.reshape(horz_filtered.shape[0])
        out_image[i, :] = horz_filtered.reshape(horz_filtered.shape[0])
        
        i += 1
    print('processed video')
    save_image(out_image, 'edges.png')
    edge_kernel = np.ones(15) / 7.5
    edge_kernel[0:4] = -edge_kernel[0:4]
    edge_kernel = edge_kernel.reshape((1, len(edge_kernel)))

    edges = convolve2d(out_image - 0.5, edge_kernel)

    #np.save('horizontal_comp.npy', horizontal_comp)
    #np.save('frames.npy', frames)
    #np.save('edges.npy', edges)
    #np.save('finish.npy', finish_image)
else:
    horizontal_comp = np.load('horizontal_comp.npy')
    frames = np.load('frames.npy')
    edges = np.load('edges.npy')
    finish_image = np.load('finish.npy')

x = []
y = []

max_finishes = []

for i in range(edges.shape[1]):
    maxes = peak_local_max(edges[:, i], min_distance=15, threshold_abs=.8)
    new_col = np.zeros(edges[:,i].shape)
    new_col[maxes] = 1
    for j in range(maxes.shape[0]):
        x.append(i)
        y.append(maxes[j,0])
        if i == finish_line:
            max_finishes.append(maxes[j,0])
    edges[:,i] = new_col
x = np.array(x)
y = np.array(y)

sample_size = 4
inlier_threshold = 8
num_inliers_threshold = 100

pil_edges = convert_to_pil(edges)
edges_draw = ImageDraw.Draw(pil_edges)

lines_drawn = 0
num_iters = 0
regs = []

while len(x) > num_inliers_threshold and lines_drawn < 10 and num_iters < 2000:
    num_iters += 1
    sample_x = np.zeros(sample_size)
    sample_y = np.zeros(sample_size)

    inds = np.random.choice(len(x), sample_size, False)
    for i in range(sample_size):
        sample_x[i] = x[inds[i]]
        sample_y[i] = y[inds[i]]
    
    reg = LinearRegression().fit(sample_x.reshape((-1, 1)), sample_y)
    predict_y = reg.predict(x.reshape((-1, 1)))
    diff = abs(y - predict_y)
    inliers = diff < inlier_threshold
    num_inliers = inliers.sum()
    #print(num_inliers)
    if num_inliers > num_inliers_threshold:
        regs.append(reg)
        outliers = diff >= inlier_threshold
        x = np.extract(outliers, x)
        y = np.extract(outliers, y)
        edges_draw.line([(0, reg.predict(np.array([[0]]))[0]), (edges.shape[1] - 1, reg.predict(np.array([[edges.shape[1] - 1]]))[0])], fill = (255, 255, 0))
        lines_drawn += 1

finish_pil = convert_to_pil(finish_image)
finish_draw = ImageDraw.Draw(finish_pil)

for count, reg in enumerate(regs):
    chest = int(reg.predict(np.array([[finish_line]]))[0])
    finish_draw.line([(chest, 0), (chest, finish_image.shape[0])], fill=(255, 0, 0))

    get_number_image(frames[chest,:,:], frames[0,:,:], finish_line, horizontal_comp[chest,:].reshape(horizontal_comp[chest,:].shape[0]), 'finish_' + str(count) + '.png')

#for finish in max_finishes:
    #finish_draw.line([(finish, 0), (finish, finish_image.shape[0])], fill=(255, 255, 0))

finish_pil.save('finish.png')


pil_edges.save('edges2.png')

"""pil_image = convert_to_pil(out_image)
id = ImageDraw.Draw(pil_image)
id.line([(finish_line, 0), (finish_line, pil_image.height)], fill=(255, 0, 0))
pil_image.save('out2.png')
print('saved image')"""
