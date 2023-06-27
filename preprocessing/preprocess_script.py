import os
from mediapipe_lips import preprocess_video

EXTENSION = '.mp4'

def preprocess_data(line, folder, path):
    os.chdir(path)
    # Account for the test shenanigans
    line = line.split()[0]

    # Get inside the dataset folder
    split_ = line.split('/')
    video_folder = split_[0]
    video = split_[1]

    video_path = os.path.join(folder, video_folder, video)
    preprocess_video(src_filename=f'{video_path}{EXTENSION}', dst_filename=f'{video_path}_lips.mp4')
    
