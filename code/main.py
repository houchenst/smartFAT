import numpy as np
import av
import matplotlib.pyplot as plt
import cv2
from skimage.color import rgb2gray
from PIL import Image

video = cv2.VideoCapture('../data/finish-line/20190413_133413.mp4')
container = av.open('../data/finish-line/20190413_133413.mp4')

num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

out_image = 0

finish_line = 700

i = 0
for frame in container.decode(video=0):
    image = rgb2gray(np.array(frame.to_image()))
    if i == 0:
        out_image = np.zeros((image.shape[1], num_frames))
    out_image[:, i] = image[finish_line:finish_line + 1,:]
    out_image[:, i] = np.zeros(out_image.shape[0]) + .5
    i += 1
plt.imshow(out_image)
plt.show()
Image.fromarray(out_image).convert('L').save('out.png')