import os
import glob
import cv2
import numpy as np
from argparse import ArgumentParser
from tools.visualization_tools import frames_to_video

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--image_format", type=str, required=True)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    image_paths = glob.glob(args.image_dir + '/*.' + args.image_format)
    image_paths.sort()
    frame_list = []
    for image_path in image_paths:
        frame = cv2.imread(image_path)
        frame_list.append(frame)
    video_path = args.image_dir + '.avi'
    frames_to_video(frame_list, video_path, args.fps)
