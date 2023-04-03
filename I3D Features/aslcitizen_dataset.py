import csv
import json
import math
import os
import os.path
import random
import glob

import cv2
import numpy as np
import torch
import torch.utils.data as data_utl

#loads rgb frames from video path, centering and downsizing as needed
def load_rgb_frames_from_video(video_path, max_frames=64, resize=(256, 256)):
    vidcap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)

    # Adjust FPS dynamically based on length of video
    frameskip = 1
    if total_frames >= 96:
        frameskip = 2
    if total_frames >= 160:
        frameskip = 3

    # Set start frame so the video is "centered" across frames
    if frameskip == 3:
        start = np.clip(int((total_frames - 192) // 2), 0, 160)
    elif frameskip == 2:
        start = np.clip(int((total_frames - 128) // 2), 0, 96)
    else:
        start = np.clip(int((total_frames - 64) // 2), 0, 64)

    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start)

    for offset in range(0, min(max_frames * frameskip, int(total_frames - start))):
        success, img = vidcap.read()
        
        if offset % frameskip == 0:
            w, h, c = img.shape
            if w < 226 or h < 226:
                d = 226. - min(w, h)
                sc = 1 + d / min(w, h)
                img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
            if w > 256 or h > 256:
                img = cv2.resize(img, (math.ceil(w * (256 / w)), math.ceil(h * (256 / h))))
            img = (img / 255.) * 2 - 1
            frames.append(img)
    return np.asarray(frames, dtype=np.float32)

def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)

    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))


class ASLCitizen(data_utl.Dataset):
    def __init__(self, datadir, transforms, video_file=None):
        self.transforms = transforms
        self.video_paths = []
        self.video_info = []

        #parse data csv
        with open(video_file, 'r') as file:
            reader = csv.reader(file)
            next(reader, None)
            for row in reader:
                fname = row[1]

                self.video_paths.append(datadir + fname)
                self.video_info.append(row)
    
    def __getitem__(self, index):
        video_path = self.video_paths[index]

        total_frames = 64
        imgs = load_rgb_frames_from_video(video_path, total_frames)
        name = self.video_info[index]

        name_dict = {'user': name[0], 'filename': name[1], 'gloss':name[2]}

        imgs = self.pad(imgs, total_frames)
        imgs = self.transforms(imgs)
        ret_img = video_to_tensor(imgs)

        return ret_img, name_dict

    def __len__(self):
        return len(self.video_paths)

    def pad(self, imgs, total_frames):
        if imgs.shape[0] < total_frames:
            num_padding = total_frames - imgs.shape[0]
            if num_padding: 
                prob = np.random.random_sample()
                if prob > 0.5: #pad with first frame
                    pad_img = imgs[0]
                    pad = np.tile(np.expand_dims(pad_img, axis=0), (num_padding, 1, 1, 1))
                    padded_imgs = np.concatenate([imgs, pad], axis=0)
                else: #pad with last frame
                    pad_img = imgs[-1]
                    pad = np.tile(np.expand_dims(pad_img, axis=0), (num_padding, 1, 1, 1))
                    padded_imgs = np.concatenate([imgs, pad], axis=0)
        else:
            padded_imgs = imgs
        return padded_imgs

    @staticmethod
    def pad_wrap(imgs, label, total_frames):
        if imgs.shape[0] < total_frames:
            num_padding = total_frames - imgs.shape[0]

            if num_padding:
                pad = imgs[:min(num_padding, imgs.shape[0])]
                k = num_padding // imgs.shape[0]
                tail = num_padding % imgs.shape[0]

                pad2 = imgs[:tail]
                if k > 0:
                    pad1 = np.array(k * [pad])[0]

                    padded_imgs = np.concatenate([imgs, pad1, pad2], axis=0)
                else:
                    padded_imgs = np.concatenate([imgs, pad2], axis=0)
        else:
            padded_imgs = imgs

        label = label[:, 0]
        label = np.tile(label, (total_frames, 1)).transpose((1, 0))

        return padded_imgs, label
