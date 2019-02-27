
# coding: utf-8

import cv2
print(cv2.__version__)

import os
import numpy as np

import sys

# ### Split MP4 Video into Individual Frames (1 Frame per Second)

frames_folder = 'frames_1ps'

try:
    if not os.path.exists(f'./s3data/{frames_folder}'):
        os.makedirs(f'./s3data/{frames_folder}')
except OSError:
    print('Error: Creating directory of frames')

vidcap = cv2.VideoCapture('./s3data/DC_to_Seattle.mp4')
fps = int(vidcap.get(cv2.CAP_PROP_FPS))
frame_no = 0
success, image = vidcap.read()
while success:
    if frame_no % fps == 0:
        name = f'./s3data/{frames_folder}/frame' + str(frame_no // fps) + '.jpg'
        cv2.imwrite(name, image)
    success, image = vidcap.read()
    frame_no += 1
vidcap.release() #closes video file or capturing device
