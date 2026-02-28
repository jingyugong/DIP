import numpy as np
import cv2

def frames_to_video(frame_list, save_path, fps=30):
    size = (frame_list[0].shape[1], frame_list[0].shape[0])
    if save_path[-3:] == 'avi':
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    elif save_path[-3:] == 'mp4':
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    VideoWriter = cv2.VideoWriter(save_path, fourcc, fps, size)
    for i, frame in enumerate(frame_list):
        VideoWriter.write(frame.astype(np.uint8))
    VideoWriter.release()
    return
