import math
import os
import argparse

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

import torch
import torch.nn
import torch.optim as optim
import numpy as np

from architecture.st_gcn import STGCN
from architecture.fc import FC
from architecture.network import Network
from asl_citizen_dataset_pose import ASLCitizen as Dataset

from tqdm import tqdm
from matplotlib import pyplot as plt
import pose_transforms

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
torch.set_default_dtype(torch.float64)

#Update files and paths as needed
video_base_path = '../data/poses/'
train_file = '../data_csv/aslcitizen_training_set.csv'
test_file = '../data_csv/aslcitzen_test_set.csv'
#Update names according to experiment number
tag = 'experiment1b'
dataset_name = "training_full"

train_transforms = pose_transforms.Compose([pose_transforms.ShearTransform(0.1),
                                            pose_transforms.RotatationTransform(0.1)])
#load data
train_ds = Dataset(datadir=video_base_path, video_file=train_file, transforms=train_transforms, pose_map_file = "pose_mapping_train.csv")
test_ds = Dataset(datadir=video_base_path, video_file=test_file, gloss_dict=train_ds.gloss_dict, pose_map_file = "pose_mapping_test.csv")
n_classes = len(train_ds.gloss_dict)


test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
gloss2idx = train_ds.gloss_dict
idx2gloss = {}
for g in gloss2idx:
    idx = gloss2idx[g]
    idx2gloss[idx] = g

#load model
n_features = 256
graph_args = {'num_nodes': 27, 'center': 0,
              'inward_edges': [[2, 0], [1, 0], [0, 3], [0, 4], [3, 5],
                               [4, 6], [5, 7], [6, 17], [7, 8], [7, 9],
                               [9, 10], [7, 11], [11, 12], [7, 13], [13, 14],
                               [7, 15], [15, 16], [17, 18], [17, 19], [19, 20],
                               [17, 21], [21, 22], [17, 23], [23, 24], [17, 25], [25, 26]]}
stgcn = STGCN(in_channels=2, graph_args=graph_args, edge_importance_weighting=True)
fc = FC(n_features=n_features, num_class=n_classes, dropout_ratio=0.05)
pose_model = Network(encoder=stgcn, decoder=fc)
pose_model.load_state_dict(torch.load('./saved_weights/_pose087561_0.665826.pt'))
pose_model.cuda()

count_total = 0
count_correct = [0, 0, 0, 0, 0, 0]
conf_matrix = np.zeros((len(gloss2idx), len(gloss2idx)))
gloss_count = np.zeros(len(gloss2idx))
user_stats = {}
user_counts = {}

# train it
pose_model.train(False)  # Set model to evaluate mode
for data in tqdm(test_loader):

    # inputs, labels, vid, src = data
    inputs, name, labels = data

    # wrap them in Variable
    inputs = inputs.cuda()
    labels = labels.cuda()
    users = name['user']
    predictions = pose_model(inputs)

    y_pred_tag = torch.softmax(predictions, dim=1)
    pred_args = torch.argsort(y_pred_tag, dim=1, descending=True)
    true_args = torch.argmax(labels, dim=1)

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
