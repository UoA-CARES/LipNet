import cv2
import os
import numpy as np

from torch import is_tensor
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

class FelixLRS2Dataset(Dataset):
    """LRS2 dataset"""
    
    def __init__(self, alignment_file, root_dir, frames_length = 1000, tokens_length = 10000, transform=None):
        """
        Args:
            alignment_file (str): Path to the txt file for the current split.
            root_dir (str): Directory for the current split.
            frames_length (int): Total number of frames for padding purposes.
            tokens_length (int): Total number of tokens for padding purposes.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        with open(alignment_file, 'r') as f:
            lines = f.readlines()
            
        # Remove the letter ending in test.txt
        self.clips = [line.split()[0] for line in lines]
        
        self.root_dir = root_dir
        self.frames_length = frames_length
        self.tokens_length = tokens_length
        self.transform = transform
        
        # Create our vocab list
        vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
        self.char_to_num = LabelEncoder()
        self.char_to_num.fit(vocab)
        
    def __len__(self):
        return len(self.clips)
    
    def load_video(self, path):
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
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)

        cap.release()

        # Normalise
        frames = np.stack(frames)
        mean = np.mean(frames.astype(np.float32))
        std = np.std(frames.astype(np.float32))
        normalized_frames = (frames.astype(np.float32) - mean) / std
    
        return normalized_frames
    
    def load_alignments(self, path): 
        """ Loads the alignments and tokenizes them.
        Args:
            path (str): Path to the alignment file.

        Returns:
            np.ndarray: Tokens in a tensor.
        """
        with open(path, 'r') as f: 
            line = f.readline() 

        line = line.lower()

        tokens = []
        words = line.split()

        for word in words:
            # Ignore start string 'text:'
            if word != 'text:': 
                tokens = [*tokens,' ', word]

        char_list = [char for string in tokens for char in string]

        return self.char_to_num.transform(char_list)[1:]
    
    def get_dir_filename(self, path):
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
    
    
    def add_padding_to_video(self, video_array):
        """Returns the video after it is padded to meet the desired number of frames. If the
        video is larger than the desired number of frames, frame sampling is used.
        
        Args:
        video_array (np.ndarray): Array containing the video frames.
        Returns:
        np.ndarray: Frames of the video with padding
        """
        num_frames = video_array.shape[0]

        if num_frames > self.frames_length:
            padding_needed = num_frames - self.frames_length
            # Downsample the outlier video to fit the desired length
            downsampling_factor = num_frames // self.frames_length
            downsampled_video_array = video_array[::downsampling_factor]
            padded_video_array = np.concatenate((downsampled_video_array, np.zeros((padding_needed,) +
                                                                                   video_array.shape[1:])), axis=0)
        else:
            padding_needed = self.frames_length - num_frames
            # Regular padding with zeros
            padding_frames = np.zeros((padding_needed,) + video_array.shape[1:], dtype=video_array.dtype)
            padded_video_array = np.concatenate((video_array, padding_frames), axis=0)

        return padded_video_array
    
    def add_padding_to_tokens(self, token_array, pad_value=0):
        """Returns the tokens array after it is padded to meet the desired number of tokens.
        Args:
            token_array (np.ndarray): Array containing the tokens
        Returns:
            np.ndarray: Tokens with padding applied.
        """
        current_length = token_array.shape[0]

        assert current_length <= self.tokens_length

        padding_needed = self.tokens_length - current_length

        # Pad the array with the specified value
        padded_array = np.pad(token_array, (0, padding_needed), mode='constant', constant_values=pad_value)

        return padded_array
    
    def load_data(self, path):
        """Returns the loaded frames and alignements using video path.
        Args:
            path (str): The path to the alignment text file without the extension.
        Returns:
            np.ndarray: Frames of the video.
            np.ndarray: Tokens for the video.
        """
        dirs, file_name = self.get_dir_filename(path)

        alignment_path = os.path.join(self.root_dir, *dirs, f'{file_name}.txt')
        video_path = os.path.join(self.root_dir, *dirs, f'{file_name}_lips.mp4')
        frames = self.load_video(video_path) 
        alignments = self.load_alignments(alignment_path)
        
        # Add padding
        frames = self.add_padding_to_video(frames)
        alignments = self.add_padding_to_tokens(alignments)

        return frames, alignments
    
    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()
            
        clip_path = self.clips[idx]
        frames, alignments = self.load_data(clip_path)
        
        if self.transform:
            frames = self.transform(frames)
            
        return frames, alignments