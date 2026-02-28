import os
import glob
import cv2
import imageio
 
def create_gif(avi_path, gif_path):
    # 读取AVI文件中的所有帧
    videos = imageio.get_reader(avi_path, 'ffmpeg')
    frame_downsample = 5
    frames = [frame for i, frame in enumerate(videos) if i % frame_downsample == 0]
    frames = [cv2.resize(frame, (288, 162)) for frame in frames]
    frames = frames * 4
 
    # 将帧保存为GIF
    imageio.mimsave(gif_path, frames, 'GIF', fps=videos.get_meta_data()['fps']//frame_downsample*0.75)
    return
 

if __name__ == "__main__":
    src_dir = "/home/gongjingyu/gcode/RGBD/code/guided-motion-diffusion/save/visualization_results/visualization_videos_filtered" 
    tgt_dir = "/home/gongjingyu/gcode/RGBD/code/guided-motion-diffusion/save/visualization_results/visualization_videos_filtered_gif" 
    src_fmt = "avi"
    tgt_fmt = "gif"
    matched_pattern = "*/*/*." + src_fmt
    matched_file_pattern = os.path.join(src_dir, matched_pattern)
    for src_file in sorted(glob.glob(matched_file_pattern)):
        tgt_file = os.path.join(tgt_dir, *src_file.split("/")[-3:]).replace(src_fmt, tgt_fmt)
        if not os.path.exists(os.path.dirname(tgt_file)):
            os.makedirs(os.path.dirname(tgt_file))
        create_gif(src_file, tgt_file)
