import math
import os
import argparse

import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import torch.nn as nn

from torchvision import transforms
import videotransforms

import numpy as np

import torch.nn.functional as F
from pytorch_i3d import InceptionI3d

from aslcitizen_dataset import ASLCitizen as Dataset
import cv2
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('data_path', type=str,
                    help='Data path')
parser.add_argument('metadata_path', type=str,
                    help='Metadata path')
parser.add_argument('outname', type=str,
                    help='Output file name')

args = parser.parse_args()
data_path = args.data_path
metadata_path = args.metadata_path
outname = args.outname


test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

dataset = Dataset(data_path, test_transforms, metadata_path)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2,
                                             pin_memory=False)

#Kinetics
#i3d = InceptionI3d(400, in_channels=3)
#i3d.load_state_dict(torch.load('rgb_imagenet.pt'))

#WLASL
#i3d = InceptionI3d(400, in_channels=3)
#i3d.replace_logits(2000)
#i3d.load_state_dict(torch.load('FINAL_nslt_2000_iters=5104_top1=32.48_top5=57.31_top10=66.31.pt'))

#ASLCitizen
i3d = InceptionI3d(400, in_channels=3)
i3d.replace_logits(2731)
i3d.load_state_dict(torch.load('../I3D/saved_weights_may/_v174_0.741945.pt'))

i3d.remove_last()
i3d.cuda()
i3d.eval()

print(outname)
output = open(outname, "w")

for data in tqdm(dataloader):
    inputs, name = data  # inputs: b, c, t, h, w
    inputs = inputs.cuda()
    name = name['gloss'][0]
    features = i3d.extract_features(inputs).cpu().detach().numpy()
    features = np.average(np.squeeze(features), axis=1)

    output.write(str(name))
    for f in features:
        output.write("\t" + str(f))
    output.write("\n")

output.close()
