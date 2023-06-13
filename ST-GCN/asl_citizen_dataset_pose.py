import csv
import glob
import numpy as np
import torch
import torch.utils.data as data_utl

#downsamples set of frames to get max frames
def downsample(frames, max_frames):
    length = frames.shape[0]
    # Adjust FPS dynamically based on length of video
    increment = max_frames / length
    if increment > 1.0:
        increment = 1.0
    curr_increment = 0
    curr_frame = 0
    new_frames = []
    for f in frames:
        curr_increment += increment
        if curr_increment > curr_frame:
            curr_frame += 1
            new_frames.append(f)
    if len(new_frames) > max_frames:
        new_frames = new_frames[:max_frames]
    return np.array(new_frames)


class ASLCitizen(data_utl.Dataset):
    def __init__(self, datadir, video_file=None, gloss_dict=None, transforms=None, pose_map_file=None):
        self.max_frames = 128
        self.transforms = transforms
        self.pose_paths = []
        self.video_info = []
        self.labels = []

        if not gloss_dict: #initialize gloss dict if not passed in as argument
            self.gloss_dict = {}
            g_count = 0

            gloss_list = []
            with open(video_file, 'r') as file:
                reader = csv.reader(file)
                next(reader, None)
                for row in reader:
                    g = row[2].strip()
                    if g not in gloss_list:
                        gloss_list.append(g)
            gloss_list.sort()

            for i in range(len(gloss_list)):
                g = gloss_list[i]
                ind = i
                self.gloss_dict[g] = ind
        else:
            self.gloss_dict = gloss_dict
            g_count = len(gloss_dict)
        
        vid_to_pose = {}
        with open(pose_map_file, "r") as file:
            reader = csv.reader(file)
            for row in reader:
                vid_fname = row[0]
                pose_fname = row[1]
                vid_to_pose[vid_fname] = pose_fname

        #parse data csv
        with open(video_file, 'r') as file:
            reader = csv.reader(file)
            next(reader, None)
            for row in reader:
                vid_fname = row[1]
                pose_fname = vid_to_pose[vid_fname]

                self.pose_paths.append(datadir + pose_fname)
                self.video_info.append(row)

                g = row[2].strip()
                self.labels.append(self.gloss_dict[g])

    def __getitem__(self, index):
        pose_path = self.pose_paths[index]

        #one-hot encode label
        l = self.labels[index]
        label = np.zeros(len(self.gloss_dict))
        label[l] = 1

        #load frames and downsample / pad as needed
        data0 = np.load(pose_path)
        length = data0.shape[0]
        if length > self.max_frames:
            data0 = downsample(data0, self.max_frames)
        if length < self.max_frames:
            data0 = np.pad(data0, ((0, self.max_frames - length), (0, 0), (0, 0)))

        #normalize keypoints using distance between shoulders as reference
        shoulder_l = data0[:, 11, :]
        shoulder_r = data0[:, 12, :]

        center = np.zeros(2)
        for i in range(len(shoulder_l)):
            center_i = (shoulder_r[i] + shoulder_l[i]) / 2
            center = center + center_i
        center = center / shoulder_l.shape[0]

        mean_dist = np.mean(np.sqrt(((shoulder_l - shoulder_r) ** 2).sum(-1)))
        if mean_dist != 0:
            scale = 1.0 / mean_dist
            data0 = data0 - center
            data0 = data0 * scale

        #select subset of keypoints for graph
        keypoints = [0, 2, 5, 11, 12, 13, 14, 33, 37, 38, 41, 42, 45, 46, 49, 50, 53, 54,
                     58, 59, 62, 63, 66, 67, 70, 71, 74]
        data0 = data0[:, 0:75, :]
        posedata = data0[:, 0:33, :]
        rhdata = data0[:, 33:54, :]
        lhdata = data0[:, 54:, :]

        data = np.concatenate([posedata, lhdata, rhdata], axis=1)
        data = data[:, keypoints, :]
        data = np.transpose(data, (2, 0, 1))

        ret_img = torch.from_numpy(data)
        if self.transforms:
            ret_img = self.transforms(ret_img)

        name = self.video_info[index]
        name_dict = {'user': name[0], 'filename': name[1], 'gloss': name[2]}

        return ret_img.double(), name_dict, torch.tensor(label, dtype=torch.float)

    def __len__(self):
        return len(self.pose_paths)
