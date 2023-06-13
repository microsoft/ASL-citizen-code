import math
import os
import argparse

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, MultiStepLR

from torchvision import transforms
import videotransforms
import numpy as np
import torch.nn.functional as F

from pytorch_i3d import InceptionI3d
from aslcitizen_dataset import ASLCitizen as Dataset
import cv2

from tqdm import tqdm
from operator import add


#Given a sorted output from the model aka ranked list, returns 
#rank of ground truth and list of other metrics
#The different indices correspond to [DCG, Top-1 Acc, Top-5 Acc, Top-10 Acc, Top-20 Acc, MRR]
def eval_metrics(sortedArgs, label):
    res, = np.where(sortedArgs == label)
    dcg = 1 / math.log2(res[0] + 1 + 1) #res values start from 0
    mrr = 1 / (res[0] + 1) 
    if res < 1:
        return res[0], [dcg, 1, 1, 1, 1, mrr]
    elif res < 5:
        return res[0], [dcg, 0, 1, 1, 1, mrr]
    elif res < 10:
        return res[0], [dcg, 0, 0, 1, 1, mrr]
    elif res < 20:
        return res[0], [dcg, 0, 0, 0, 1, mrr]
    else:
        return res[0], [dcg, 0, 0, 0, 0, mrr]

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                       videotransforms.RandomHorizontalFlip()])
test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

#Update files and paths as needed
video_base_path = '../../final_dataset/ASL_Citizen/videos/'
train_file = '../../final_dataset/ASL_Citizen/splits/train.csv'
test_file = '../../final_dataset/ASL_Citizen/splits/test.csv'
#Update names according to experiment number
tag = 'may'
dataset_name = "v1"

#load data
train_ds = Dataset(datadir=video_base_path, transforms=train_transforms, video_file=train_file)
print(len(train_ds.gloss_dict))

test_ds = Dataset(datadir=video_base_path, transforms=test_transforms, video_file=test_file, gloss_dict=train_ds.gloss_dict)
print(len(test_ds.gloss_dict))

#train_loader = torch.utils.data.DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)

gloss2idx = train_ds.gloss_dict
idx2gloss = {}
for g in gloss2idx:
    idx = gloss2idx[g]
    idx2gloss[idx] = g

# Load i3d model
i3d = InceptionI3d(400, in_channels=3)
i3d.replace_logits(2731)
#Update model weights here
i3d.load_state_dict(torch.load('./saved_weights_may/_v174_0.741945.pt'))
i3d.cuda()

#For wlasl
#i3d.replace_logits(2000)
#i3d.load_state_dict(torch.load('FINAL_nslt_2000_iters=5104_top1=32.48_top5=57.31_top10=66.31.pt'))
#i3d.cuda()

count_total = 0
#the different indices correspond to sum [DCG, Top-1, Top-5, Top-10, Top-20, MRR]
count_correct = [0, 0, 0, 0, 0, 0]
conf_matrix = np.zeros((len(gloss2idx), len(gloss2idx)))
gloss_count = np.zeros(len(gloss2idx))
user_stats = {}
user_counts = {}

i3d.train(False)  # Set model to evaluate mode
for data in tqdm(test_loader):

    # inputs, labels, vid, src = data
    inputs, name, labels = data
    #print(name)

    # wrap them in Variable
    inputs = inputs.cuda()
    t = inputs.size(2)
    labels = labels.cuda()
    users = name['user']

    per_frame_logits = i3d(inputs, pretrained=False)
    ground_truth = torch.max(labels, dim=2)[0]

    # upsample to input size
    per_frame_logits = F.upsample(per_frame_logits, t, mode='linear')

    predictions = torch.max(per_frame_logits, dim=2)[0]
    y_pred_tag = torch.softmax(predictions, dim=1)
    pred_args = torch.argsort(y_pred_tag, dim=1, descending=True)
    true_args = torch.argmax(ground_truth, dim=1)

    for i in range(len(pred_args)):
        pred = pred_args[i].cpu().numpy()
        gti = true_args[i].cpu().numpy()

        res, counts = eval_metrics(pred, gti)

        #update overall metrics counts
        count_correct = list(map(add, counts, count_correct))
        count_total = count_total + 1

        #update confusion metrics counts
        conf_matrix[gti, pred[0]] =  conf_matrix[gti, pred[0]] + 1
        gloss_count[gti] = gloss_count[gti] + 1

        #update user metrics counts
        u = users[i]
        if u not in user_counts:
            user_counts[u] = 1
            user_stats[u] = counts
        else:
            user_counts[u] = user_counts[u] + 1
            user_stats[u] = list(map(add, counts, user_stats[u]))

#output overall metrics
with open('output ' + tag + '.txt', 'w') as f:
    f.write("Total files in eval = " + str(count_total) + '\n')
    f.write("Discounted Cumulative Gain is " + str(count_correct[0]/count_total)+ '\n')
    f.write("Mean Reciprocal Rank is " + str(count_correct[5]/count_total)+ '\n')
    f.write("Top-1 accuracy is " + str(count_correct[1]/count_total)+ '\n')
    f.write("Top-5 accuracy is " + str(count_correct[2]/count_total)+ '\n')
    f.write("Top-10 accuracy is " + str(count_correct[3]/count_total)+ '\n')
    f.write("Top-20 accuracy is " + str(count_correct[4]/count_total)+ '\n')
    f.write('\n')

#output user stats
with open('user_stats ' + tag + '.txt', 'w') as f:
   for u in user_counts:
       f.write("User: " + u + '\n')
       f.write("Files: " + str(user_counts[u]) + '\n')
       f.write("Discounted Cumulative Gain is " + str(user_stats[u][0]/user_counts[u])+ '\n')
       f.write("Mean Reciprocal Rank is " + str(user_stats[u][5]/user_counts[u])+ '\n')
       f.write("Top-1 accuracy is " + str(user_stats[u][1]/user_counts[u])+ '\n')
       f.write("Top-5 accuracy is " + str(user_stats[u][2]/user_counts[u])+ '\n')
       f.write("Top-10 accuracy is " + str(user_stats[u][3]/user_counts[u])+ '\n')
       f.write("Top-20 accuracy is " + str(user_stats[u][4]/user_counts[u])+ '\n')
       f.write('\n')

#output complete confusion matrix
np.savetxt('confusion matrix ' + tag + '.txt', conf_matrix, fmt='%d')

#output mini confusion matrix
with open('conf_mini_' + tag + '.csv', 'w') as f:
    for i in range(len(idx2gloss)):
        g = idx2gloss[i]

        counts = conf_matrix[i]
        if np.sum(counts) != 0:
            acc = conf_matrix[i, i] / np.sum(counts)
        else:
            acc = 0
        sortedArgs = counts.argsort()[::-1][:5]

        pred0 = idx2gloss[sortedArgs[0]]
        count0 = conf_matrix[i, sortedArgs[0]]

        pred1 = idx2gloss[sortedArgs[1]]
        count1 = conf_matrix[i, sortedArgs[1]]

        pred2 = idx2gloss[sortedArgs[2]]
        count2 = conf_matrix[i, sortedArgs[2]]

        pred3 = idx2gloss[sortedArgs[3]]
        count3 = conf_matrix[i, sortedArgs[3]]

        pred4 = idx2gloss[sortedArgs[4]]
        count4 = conf_matrix[i, sortedArgs[4]]

        f.write(g + "," + str(acc) + "," + pred0 + "," + str(count0) + "," + pred1 + "," + str(count1) + "," + pred2 + "," + str(count2)+ "," + pred3 + "," + str(count3)+ "," + pred4 + "," + str(count4) + '\n')

