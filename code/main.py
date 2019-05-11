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

# convert numpy arraw to pillow image
def convert_to_pil(arr):
    arr = arr * 256
    return Image.fromarray(arr).convert('RGB')

# save a numpy array as an image
def save_image(arr, path):
    convert_to_pil(arr).save(path)

# gets and saves a small image that contains the hip number for the runner that is finishing in frame.
# args:
# frame: frame of the runner finishing
# base_image: background image
# finish_line: x-coordinate of the finish line
# horz_comp: vertically filtered row showing where the runner is horizontally in the frame
# file_name: file name to store the image as
def get_number_image(frame, base_image, finish_line, horz_comp, file_name):
    horz_comp2 = gaussian_filter1d(horz_comp, sigma=5) # blur to eliminate little gaps

    # find where there stops being runner and use that as the lower bound of the horizontal slice
    lower_bound = finish_line - 20
    for i in range(finish_line - 20, 0, -1):
        if horz_comp2[i] < .1:
            lower_bound = i
            break
    # get subtracted image
    sub = abs(frame - base_image)
    
    # take the vertical stripe of the image where there is runner
    segment = np.transpose(frame[lower_bound:finish_line+10, :])

    # take the vertical stripe of the difference image
    segment_filter = np.transpose(sub[lower_bound:finish_line+10, :])

    # make a horizontal filter to convolve with the difference segment
    horz_filter = np.ones((1, segment.shape[1])) / segment.shape[1]
    
    # make a column of pixels whose values are 1 where there is a runner and 0 where there isn't
    horz_filtered = convolve2d(segment_filter, horz_filter, mode='valid').reshape(segment.shape[0])
    horz_filtered_g = horz_filtered > 0.05
    horz_filtered_l = horz_filtered <= 0.05
    horz_filtered[horz_filtered_g] = 1
    horz_filtered[horz_filtered_l] = 0

    # calculate the height of the horizontal stripe we are going to extract from the vertical stripe
    segment_height = int(np.sum(horz_filtered))

    # extract rows where horz_filtered = 1 and add them to output
    output = np.zeros((segment_height, segment.shape[1]))
    cntr = 0
    for i in range(len(horz_filtered)):
        if horz_filtered[i] == 1:
            output[cntr,:] = segment[i,:]
            cntr += 1

    # take only the middle 40% of the image (vertically), because the hip is in the middle of a person
    output = output[int(.3 * output.shape[0]):int(.7 * output.shape[0]),:]

    # flip the image because it got flipped somehow
    output = np.flip(output, axis=1)

    save_image(output, file_name)

finish_line = 360 # x coordinate of the finish line

# these variables are initialized when the first frame of the video is processed
frames = 0 # numpy array of every frame
horizontal_comp = 0 # array of every vert_filter-convolved row
edges = 0 # array of the edges to use RANSAC line-fitting on
finish_image = 0 # cross-sectional image

# array saving and loading was used to speed up development and can be enabled by removing the 'True or' at the beginning of the if condition
if True or not os.path.isfile('edges.npy') or not os.path.isfile('finish.npy') or not os.path.isfile('horizontal_comp.npy') or not os.path.isfile('frames.npy'):
    path = '../data/finish-line/20190413_134043.mp4'
    video = cv2.VideoCapture(path)
    container = av.open(path)

    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # these variables are initialized when processing the first frame
    base_image = 0 # background image
    vert_filter = 0 # vertical filter for convolving down to a single row

    print('loaded video')
    i = 0 # frame counter
    

    for frame in container.decode(video=0):
        # convert image to grayscale
        image = rgb2gray(np.array(frame.to_image()))
        if i == 0:

            # initialize arrays and vertical filter
            finish_image = np.zeros((image.shape[1], num_frames))
            base_image = image
            vert_filter = np.ones((1, image.shape[1])) / image.shape[1]
            # taper the filter towards the bottom
            for j in range(int(vert_filter.shape[1] * 2 / 3), vert_filter.shape[1]):
                vert_filter[0, j] = vert_filter[0, j] * (1 - 2 * (j - vert_filter.shape[1] * 2 / 3) / vert_filter.shape[1])

            frames = np.zeros((num_frames, image.shape[0], image.shape[1]))
            horizontal_comp = np.zeros((num_frames, image.shape[0]))
        finish_image[:, i] = image[finish_line, :] # take vertical strip of image at finish line
        frames[i,:,:] = image 
        image = abs(image - base_image) # subtract out background
        
        horz_filtered = convolve2d(image, vert_filter, mode='valid') # convolve with vertical filter

        # clamp values to be 0 or 1
        horz_bg = horz_filtered < .03
        horz_person = horz_filtered > .03
        horz_filtered[horz_bg] = 0
        horz_filtered[horz_person] = 1

        # add row to large array
        horizontal_comp[i, :] = horz_filtered.reshape(horz_filtered.shape[0])
        
        i += 1
    print('processed video')
    # kernel to edge detect the lines of the runners
    edge_kernel = np.ones(15) / 7.5
    edge_kernel[0:4] = -edge_kernel[0:4]
    edge_kernel = edge_kernel.reshape((1, len(edge_kernel)))

    # convoled image
    edges = convolve2d(horizontal_comp - 0.5, edge_kernel)

    #np.save('horizontal_comp.npy', horizontal_comp)
    #np.save('frames.npy', frames)
    #np.save('edges.npy', edges)
    #np.save('finish.npy', finish_image)
else:
    horizontal_comp = np.load('horizontal_comp.npy')
    frames = np.load('frames.npy')
    edges = np.load('edges.npy')
    finish_image = np.load('finish.npy')

# points to fit using RANSAC
x = []
y = []

# prior method for finding finishes
max_finishes = []

save_image(horizontal_comp, 'edges3.png')
save_image(edges, 'edges.png')

for i in range(edges.shape[1]):
    # for each row run non-max suppression
    maxes = peak_local_max(edges[:, i], min_distance=15, threshold_abs=.8)
    # update image to only include maxes
    new_col = np.zeros(edges[:,i].shape)
    new_col[maxes] = 1
    for j in range(maxes.shape[0]):
        # add points to data arrays
        x.append(i)
        y.append(maxes[j,0])
        if i == finish_line:
            max_finishes.append(maxes[j,0])
    edges[:,i] = new_col

x = np.array(x)
y = np.array(y)

save_image(edges, 'edges4.png')

# constants for RANSAC
sample_size = 4 # sample size of points considered on each iteration
inlier_threshold = 5 # distance a point can be from the line to be considered an inlier
num_inliers_threshold = 100 # number of inliers required to accept a line

# image to draw on
pil_edges = convert_to_pil(edges)
edges_draw = ImageDraw.Draw(pil_edges)

lines_drawn = 0
num_iters = 0
regs = [] # list of fitted lines

# i
while len(x) > num_inliers_threshold and lines_drawn < 10 and num_iters < 2000:
    num_iters += 1

    # take a random sample of points
    sample_x = np.zeros(sample_size)
    sample_y = np.zeros(sample_size)

    inds = np.random.choice(len(x), sample_size, False)
    for i in range(sample_size):
        sample_x[i] = x[inds[i]]
        sample_y[i] = y[inds[i]]
    
    # fit a line to the sample
    reg = LinearRegression().fit(sample_x.reshape((-1, 1)), sample_y)

    # the y values from the line for every x in the set of points
    predict_y = reg.predict(x.reshape((-1, 1)))

    # difference from the real y values
    diff = abs(y - predict_y)

    # find number of inliers
    inliers = diff < inlier_threshold
    num_inliers = inliers.sum()

    # if the line is acceptable
    if num_inliers > num_inliers_threshold:
        regs.append(reg) # add to set of lines
        outliers = diff >= inlier_threshold * 2.5

        # remove inliers from set of points
        x = np.extract(outliers, x)
        y = np.extract(outliers, y)

        # draw the fitted line for visualization purposes
        edges_draw.line([(0, reg.predict(np.array([[0]]))[0]), (edges.shape[1] - 1, reg.predict(np.array([[edges.shape[1] - 1]]))[0])], fill = (255, 255, 0))
        lines_drawn += 1

# cross-sectional image to draw on
finish_pil = convert_to_pil(finish_image)
finish_draw = ImageDraw.Draw(finish_pil)


# for each fitted line, draw a line in the cross-sectional image and get an image to
for count, reg in enumerate(regs):
    chest = int(reg.predict(np.array([[finish_line]]))[0])
    finish_draw.line([(chest, 0), (chest, finish_image.shape[0])], fill=(255, 0, 0))
    get_number_image(frames[chest,:,:], frames[0,:,:], finish_line, horizontal_comp[chest,:].reshape(horizontal_comp[chest,:].shape[0]), 'finish_' + str(count) + '.png')

#for finish in max_finishes:
    #finish_draw.line([(finish, 0), (finish, finish_image.shape[0])], fill=(255, 255, 0))

finish_pil.save('finish.png')
pil_edges.save('edges2.png')