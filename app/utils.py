import tensorflow as tf
import cv2
import os
import numpy as np
import imageio


from typing import List, Union

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")

# Mapping integers back to original characters
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

def load_video(path:str) -> List[float]:
    """Loads the frames of a video file, converts frames to grayscale,
    crops the mouth, and standardises the frames.
    
    Args:
        path (string): Path for the video file.
        
    Returns:
        List[float]: Frames of the video in a tensor.
    """
    cap = cv2.VideoCapture(path)
    frames = []
    
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        flag, frame = cap.read()
        
        if not flag:
            return
        
        frame = tf.image.rgb_to_grayscale(frame)
        
        # Crop using static co-ords (could use another model to get exact lips location)
        frames.append(frame[190:236, 80:220, :])
        
    cap.release()
    
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    
    return tf.cast((frames-mean), tf.float32) / std

    
def load_alignments(path: str) -> np.ndarray:
    """ Loads the alignments and tokenizes them.
    Args:
        path (str): Path to the alignment file.

    Returns:
        np.ndarray: Tokens in a tensor.
    """
    with open(path, 'r') as f:
        lines = f.readlines()

    tokens = []

    for line in lines:
        line = line.split()

        # Ignore start/end string 'sil'
        if line[2] != 'sil':
            tokens = [*tokens, ' ', line[2]]

    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]


def get_dir_filename(path: str) -> Union[np.ndarray, str]:
    """Returns a list of directories and the filename.
    Args:
        path (str): A path.

    Returns:
        np.ndarray: The directories.
        string: Filename.
    """
    path, file_name = os.path.split(path)
    directories = []

    while True:
        path, directory = os.path.split(path)
        if directory != "":
            directories.append(directory)
        else:
            if path != "":
                directories.append(path)
            break
    # Reverse the directories list
    directories = directories[::-1]

    return directories, file_name


def load_data(path: tf.Tensor) -> Union[np.ndarray, np.ndarray]:
    """Returns the loaded frames and alignements using video path.
    Args:
        path (tf.Tensor): The path to the video file.
    Returns:
        np.ndarray: Frames of the video in a tensor.
        np.ndarray: Tokens in a tensor.
        """
    path = bytes.decode(path.numpy())

    dirs, file_name = get_dir_filename(path)

    # Remove the extension
    file_name = file_name.split('.')[0]

    video_path = path
    alignment_path = os.path.join(
        'data', 'alignments', 's1', f'{file_name}.align')
    frames = load_video(video_path)
    alignments = load_alignments(alignment_path)

    return frames, alignments
