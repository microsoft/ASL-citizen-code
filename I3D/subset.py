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
from asl_citizen_dataset_with_labels import ASLCitizen as Dataset
import cv2

from tqdm import tqdm
from operator import add

def eval_metrics(sortedArgs, label):
    res, = np.where(sortedArgs == label)
    #print(res[0])
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

video_base_path = '../data/PaidDataSet/videos/'
train_file = '../data_csv/aslcitizen_training_set.csv'
test_file = '../data_csv/aslcitzen_test_set.csv'
tag = 'experiment1d'
dataset_name = "training_subset"

train_ds = Dataset(datadir=video_base_path, transforms=train_transforms, video_file=train_file)
print(len(train_ds.gloss_dict))

test_ds = Dataset(datadir=video_base_path, transforms=test_transforms, video_file=test_file, gloss_dict=train_ds.gloss_dict)
print(len(test_ds.gloss_dict))

test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)

glosses = train_ds.gloss_dict

wlasl_glosses = {}
#get wlasl gloss to index mappings
with open('../utils/wlasl_class_list.txt') as f:
    for line in f:
        line = line.strip()
        words = line.split('\t')
        wlasl_glosses[words[1]] = int(words[0])

#get wlasl glosses belonging to overlapping subset
wlasl_subset = []
with open('../utils/wlasl_subset.txt') as f:
    for line in f:
        line = line.strip()
        wlasl_subset.append(wlasl_glosses[line])

#get aslcitizen glosses belonging to overlapping subset
asl_subset = []
with open('../utils/aslcitizen_subset.txt') as f:
    for line in f:
        line = line.strip()
        asl_subset.append(glosses[line])

#get mapping between wlasl and asl_citizen subsets
asl_to_wlasl = {}
with open('../utils/aslcitizen_wlasl_subset_mapping.txt') as f:
    for line in f:
        line = line.strip()
        words = line.split('\t')
        asl_to_wlasl[words[0]] = words[1]

asl_subset.sort()
wlasl_subset.sort()

#gloss to index mapping for subset
asl_gloss_map = {} #old to new
count = 0
#for g in wlasl_subset:
for g in asl_subset:
    asl_gloss_map[g] = count
    count += 1



# Load i3d model
i3d = InceptionI3d(400, in_channels=3)

#WLASL
#i3d.replace_logits(2000)
#i3d.load_state_dict(torch.load('wlasl.pt'))

#aslcitizen
i3d.replace_logits(2731)
i3d.load_state_dict(torch.load('./saved_weights_jan_1a/_jan_1a75_0.736444.pt'))
i3d.cuda()

count_total = 0
count_correct = [0, 0, 0, 0, 0, 0]
conf_matrix = np.zeros((len(glosses), len(glosses)))
gloss_count = np.zeros(len(glosses))
user_stats = {}
user_counts = {}

# train it

i3d.train(False)  # Set model to evaluate mode
for data in tqdm(test_loader):

    # inputs, labels, vid, src = data
    inputs, name, labels = data

    # wrap them in Variable
    inputs = inputs.cuda()
    t = inputs.size(2)
    labels = labels.cuda()
    users = name['user']

    gt = torch.max(labels, dim=2)[0]
    g = name['gloss'][0].strip()

    if g in asl_to_wlasl:
#        g = asl_to_wlasl[g]
#        g_ind = wlasl_glosses[g]
        g_ind = glosses[g]

        per_frame_logits = i3d(inputs)

        # upsample to input size
        per_frame_logits = F.upsample(per_frame_logits, t, mode='linear')

        predictions = torch.max(per_frame_logits, dim=2)[0]
        predictions = predictions.squeeze()
        pred_subset = predictions[asl_subset]
#        pred_subset = predictions[wlasl_subset]
        y_pred_tag = torch.softmax(pred_subset, dim=0)
        pred_args = torch.argsort(y_pred_tag, dim=0, descending=True)

        true_args = asl_gloss_map[g_ind]
        pred = pred_args.cpu().numpy()
        gti = true_args

        res, counts = eval_metrics(pred, gti)
        count_correct = list(map(add, counts, count_correct))
        count_total = count_total + 1

        u = users[0]
        if u not in user_counts:
            user_counts[u] = 1
            user_stats[u] = counts
        else:
            user_counts[u] = user_counts[u] + 1
            user_stats[u] = list(map(add, counts, user_stats[u]))

with open('output ' + tag + '.txt', 'w') as f:
    f.write("Total files in eval = " + str(count_total) + '\n')
    f.write("Discounted Cumulative Gain is " + str(count_correct[0]/count_total)+ '\n')
    f.write("Mean Reciprocal Rank is " + str(count_correct[5]/count_total)+ '\n')
    f.write("Top-1 accuracy is " + str(count_correct[1]/count_total)+ '\n')
    f.write("Top-5 accuracy is " + str(count_correct[2]/count_total)+ '\n')
    f.write("Top-10 accuracy is " + str(count_correct[3]/count_total)+ '\n')
    f.write("Top-20 accuracy is " + str(count_correct[4]/count_total)+ '\n')
    f.write('\n')

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
