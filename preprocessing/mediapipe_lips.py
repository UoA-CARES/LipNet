#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2023 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch
import torchaudio
import torchvision
import cv2
import os
import numpy as np
import mediapipe as mp

from skimage import transform as tf

class FunctionalModule(torch.nn.Module):
    def __init__(self, functional):
        super().__init__()
        self.functional = functional

    def forward(self, input):
        return self.functional(input)


class VideoTransform:
    def __init__(self, speed_rate):
        self.video_pipeline = torch.nn.Sequential(
            FunctionalModule(lambda x: x.unsqueeze(-1)),
            FunctionalModule(lambda x: x if speed_rate == 1 else torch.index_select(x, dim=0, index=torch.linspace(0, x.shape[0]-1, int(x.shape[0] / speed_rate), dtype=torch.int64))),
            FunctionalModule(lambda x: x.permute(3, 0, 1, 2)),
            FunctionalModule(lambda x: x / 255.),
            torchvision.transforms.CenterCrop(88),
            torchvision.transforms.Normalize(0.421, 0.165),
        )

    def __call__(self, sample):
        return self.video_pipeline(sample)


class AudioTransform:
    def __init__(self):
        self.audio_pipeline = torch.nn.Sequential(
            FunctionalModule(lambda x: torch.nn.functional.layer_norm(x, x.shape, eps=0)),
            FunctionalModule(lambda x: x.transpose(0, 1)),
        )

    def __call__(self, sample):
        return self.audio_pipeline(sample)


def linear_interpolate(landmarks, start_idx, stop_idx):
    start_landmarks = landmarks[start_idx]
    stop_landmarks = landmarks[stop_idx]
    delta = stop_landmarks - start_landmarks
    for idx in range(1, stop_idx-start_idx):
        landmarks[start_idx+idx] = start_landmarks + idx/float(stop_idx-start_idx) * delta
    return landmarks


def warp_img(src, dst, img, std_size):
    tform = tf.estimate_transform('similarity', src, dst)
    warped = tf.warp(img, inverse_map=tform.inverse, output_shape=std_size)
    warped = (warped * 255).astype('uint8')
    return warped, tform


def apply_transform(transform, img, std_size):
    warped = tf.warp(img, inverse_map=transform.inverse, output_shape=std_size)
    warped = (warped * 255).astype('uint8')
    return warped


def cut_patch(img, landmarks, height, width, threshold=5):
    center_x, center_y = np.mean(landmarks, axis=0)
    # # Check for too much bias in height and width
    if abs(center_y - img.shape[0] / 2) > height + threshold:
        raise Exception('too much bias in height')
    if abs(center_x - img.shape[1] / 2) > width + threshold:
        raise Exception('too much bias in width')
    # Calculate bounding box coordinates
    y_min = int(round(np.clip(center_y - height, 0, img.shape[0])))
    y_max = int(round(np.clip(center_y + height, 0, img.shape[0])))
    x_min = int(round(np.clip(center_x - width, 0, img.shape[1])))
    x_max = int(round(np.clip(center_x + width, 0, img.shape[1])))
    # Cut the image
    cutted_img = np.copy(img[y_min:y_max, x_min:x_max])
    return cutted_img


class VideoProcess:
    def __init__(self, mean_face_path="./20words_mean_face.npy", crop_width=96, crop_height=96,
                 start_idx=3, stop_idx=4, window_margin=12, convert_gray=True):
        self.reference = np.load(os.path.join(os.path.dirname(''), mean_face_path))
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.start_idx = start_idx
        self.stop_idx = stop_idx
        self.window_margin = window_margin
        self.convert_gray = convert_gray

    def __call__(self, video, landmarks):
        # Pre-process landmarks: interpolate frames that are not detected
        preprocessed_landmarks = self.interpolate_landmarks(landmarks)
        # Exclude corner cases: no landmark in all frames
        if not preprocessed_landmarks:
            return
        # Affine transformation and crop patch
        sequence = self.crop_patch(video, preprocessed_landmarks)
        assert sequence is not None, f"cannot crop a patch from {filename}."
        return sequence


    def crop_patch(self, video, landmarks):
        sequence = []
        for frame_idx, frame in enumerate(video):
            window_margin = min(self.window_margin // 2, frame_idx, len(landmarks) - 1 - frame_idx)
            smoothed_landmarks = np.mean([landmarks[x] for x in range(frame_idx - window_margin, frame_idx + window_margin + 1)], axis=0)
            smoothed_landmarks += landmarks[frame_idx].mean(axis=0) - smoothed_landmarks.mean(axis=0)
            transformed_frame, transformed_landmarks = self.affine_transform(frame,smoothed_landmarks,self.reference,grayscale=self.convert_gray)
            patch = cut_patch(transformed_frame, transformed_landmarks[self.start_idx:self.stop_idx], self.crop_height//2, self.crop_width//2,)
            sequence.append(patch)
        return np.array(sequence)


    def interpolate_landmarks(self, landmarks):
        valid_frames_idx = [idx for idx, lm in enumerate(landmarks) if lm is not None]

        if not valid_frames_idx:
            return None

        for idx in range(1, len(valid_frames_idx)):
            if valid_frames_idx[idx] - valid_frames_idx[idx - 1] > 1:
                landmarks = linear_interpolate(landmarks, valid_frames_idx[idx - 1], valid_frames_idx[idx])

        valid_frames_idx = [idx for idx, lm in enumerate(landmarks) if lm is not None]

        # Handle corner case: keep frames at the beginning or at the end that failed to be detected
        if valid_frames_idx:
            landmarks[:valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
            landmarks[valid_frames_idx[-1]:] = [landmarks[valid_frames_idx[-1]]] * (len(landmarks) - valid_frames_idx[-1])

        assert all(lm is not None for lm in landmarks), "not every frame has landmark"

        return landmarks


    def affine_transform(self, frame, landmarks, reference, grayscale=False,
                         target_size=(256, 256), reference_size=(256, 256), stable_points=(0, 1, 2, 3),
                         interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, border_value=0):
        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        stable_reference = self.get_stable_reference(reference, reference_size, target_size)
        transform = self.estimate_affine_transform(landmarks, stable_points, stable_reference)
        transformed_frame, transformed_landmarks = self.apply_affine_transform(frame, landmarks, transform, target_size, interpolation, border_mode, border_value)

        return transformed_frame, transformed_landmarks


    def get_stable_reference(self, reference, reference_size, target_size):
        # -- right eye, left eye, nose tip, mouth center
        stable_reference = np.vstack([
            np.mean(reference[36:42], axis=0),
            np.mean(reference[42:48], axis=0),
            np.mean(reference[31:36], axis=0),
            np.mean(reference[48:68], axis=0)
        ])
        stable_reference[:, 0] -= (reference_size[0] - target_size[0]) / 2.0
        stable_reference[:, 1] -= (reference_size[1] - target_size[1]) / 2.0
        return stable_reference


    def estimate_affine_transform(self, landmarks, stable_points, stable_reference):
        return cv2.estimateAffinePartial2D(np.vstack([landmarks[x] for x in stable_points]), stable_reference, method=cv2.LMEDS)[0]


    def apply_affine_transform(self, frame, landmarks, transform, target_size, interpolation, border_mode, border_value):
        transformed_frame = cv2.warpAffine(frame, transform, dsize=(target_size[0], target_size[1]),
                                           flags=interpolation, borderMode=border_mode, borderValue=border_value)
        transformed_landmarks = np.matmul(landmarks, transform[:, :2].transpose()) + transform[:, 2].transpose()
        return transformed_frame, transformed_landmarks


class LandmarksDetector:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.short_range_detector = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=0)
        self.full_range_detector = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=1)

    def __call__(self, filename):
        video_frames = torchvision.io.read_video(filename, pts_unit='sec')[0].numpy()
        landmarks = self.detect(video_frames, self.full_range_detector)
        if all(element is None for element in landmarks):
            landmarks = self.detect(video_frames, self.short_range_detector)
            assert any(l is not None for l in landmarks), "Cannot detect any frames in the video"
        return landmarks

    def detect(self, video_frames, detector):
        landmarks = []
        for frame in video_frames:
            results = detector.process(frame)
            if not results.detections:
                landmarks.append(None)
                continue
            face_points = []
            for idx, detected_faces in enumerate(results.detections):
                max_id, max_size = 0, 0
                bboxC = detected_faces.location_data.relative_bounding_box
                ih, iw, ic = frame.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                bbox_size = (bbox[2] - bbox[0]) + (bbox[3] - bbox[1])
                if bbox_size > max_size:
                    max_id, max_size = idx, bbox_size
                lmx = [
                    [int(detected_faces.location_data.relative_keypoints[self.mp_face_detection.FaceKeyPoint(0).value].x * iw),
                     int(detected_faces.location_data.relative_keypoints[self.mp_face_detection.FaceKeyPoint(0).value].y * ih)],
                    [int(detected_faces.location_data.relative_keypoints[self.mp_face_detection.FaceKeyPoint(1).value].x * iw),
                     int(detected_faces.location_data.relative_keypoints[self.mp_face_detection.FaceKeyPoint(1).value].y * ih)],
                    [int(detected_faces.location_data.relative_keypoints[self.mp_face_detection.FaceKeyPoint(2).value].x * iw),
                     int(detected_faces.location_data.relative_keypoints[self.mp_face_detection.FaceKeyPoint(2).value].y * ih)],
                    [int(detected_faces.location_data.relative_keypoints[self.mp_face_detection.FaceKeyPoint(3).value].x * iw),
                     int(detected_faces.location_data.relative_keypoints[self.mp_face_detection.FaceKeyPoint(3).value].y * ih)],
                    ]
                face_points.append(lmx)
            landmarks.append(np.array(face_points[max_id]))
        return landmarks




class AVSRDataLoader:
    def __init__(self, modality, speed_rate=1, transform=True, detector="retinaface", convert_gray=True):
        self.modality = modality
        self.transform = transform
        if self.modality in ["audio", "audiovisual"]:
            self.audio_transform = AudioTransform()
        if self.modality in ["video", "audiovisual"]:
            if detector == "mediapipe":
                self.video_process = VideoProcess(convert_gray=convert_gray)
            self.video_transform = VideoTransform(speed_rate=speed_rate)


    def load_data(self, data_filename, landmarks=None, transform=True):
        if self.modality == "audio":
            audio, sample_rate = self.load_audio(data_filename)
            audio = self.audio_process(audio, sample_rate)
            return self.audio_transform(audio) if self.transform else audio
        if self.modality == "video":
            video = self.load_video(data_filename)
            video = self.video_process(video, landmarks)
            video = torch.tensor(video)
            return self.video_transform(video) if self.transform else video
        if self.modality == "audiovisual":
            rate_ratio = 640
            audio, sample_rate = self.load_audio(data_filename)
            audio = self.audio_process(audio, sample_rate)
            video = self.load_video(data_filename)
            video = self.video_process(video, landmarks)
            video = torch.tensor(video)
            min_t = min(len(video), audio.size(1) // rate_ratio)
            audio = audio[:, :min_t*rate_ratio]
            video = video[:min_t]
            if self.transform:
                audio = self.audio_transform(audio)
                video = self.video_transform(video)
            return video, audio


    def load_audio(self, data_filename):
        waveform, sample_rate = torchaudio.load(data_filename, normalize=True)
        return waveform, sample_rate


    def load_video(self, data_filename):
        return torchvision.io.read_video(data_filename, pts_unit='sec')[0].numpy()


    def audio_process(self, waveform, sample_rate, target_sample_rate=16000):
        if sample_rate != target_sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform

def save2vid(filename, vid, frames_per_second):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torchvision.io.write_video(filename, vid, frames_per_second)

def preprocess_video(src_filename, dst_filename):
    landmarks = landmarks_detector(src_filename)
    data = dataloader.load_data(src_filename, landmarks)
    fps = 25 # cv2.VideoCapture(src_filename).get(cv2.CAP_PROP_FPS)
    save2vid(dst_filename, data, fps)
    torchvision.io.write_video(dst_filename, data, fps)
    return

dataloader = AVSRDataLoader(modality="video", speed_rate=1, transform=False, detector="mediapipe", convert_gray=False)
landmarks_detector = LandmarksDetector()