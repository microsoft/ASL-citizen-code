import math
import os
import argparse
import numpy as np
import csv
from scipy.spatial.distance import minkowski, cosine
import math

import matplotlib.pyplot as plt

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
from operator import add

from timeit import default_timer as timer

parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('seed_file', type=str,
                    help='Seed File')
parser.add_argument('test_file', type=str,
                    help='Test File')
parser.add_argument('output_name', type=str,
                    help='Output file name')

args = parser.parse_args()
seed_file = args.seed_file
test_file = args.test_file
output_name = args.output_name

#to load data from tsv file
def load_data(file):
    print(file)
    with open(file) as f:
        ncols = len(f.readline().split('\t'))
    data = np.loadtxt(file, delimiter='\t', usecols=range(1,ncols))
    with open(file) as f:
        reader = csv.reader(f, delimiter="\t")
        labels = []
        g_index = 0
        g_dict = {}
        for row in reader:
            label = row[0].strip()
            labels.append(label) 

            if label not in g_dict:
                g_dict[label] = g_index
                g_index += 1

    return data, labels, g_dict

#returns ranked list of indices (corresponding to glosses)
#calculates the distance between a given feature embeddings 
# and all feature embeddings from training dataset
def minkowski_distance(feature, seeddata, glosses, g_dict):
    distances = [0] * len(g_dict)
    gloss_dist = {}
    for i in range(len(seeddata)):
        train_feat = seeddata[i]
        train_gloss = glosses[i]
        dist = cosine(train_feat, feature)

        #for a given gloss, record the shortest distance
        #between train data video and given video
        if train_gloss not in gloss_dist:
            gloss_dist[train_gloss] = dist
        elif dist < gloss_dist[train_gloss]:
            gloss_dist[train_gloss] = dist

    for g in gloss_dist:
        index = g_dict[g]
        distances[index] = gloss_dist[g]

    sortedargs = np.argsort(distances)
    return sortedargs


#Given a sorted output from the model aka ranked list, returns 
#rank of ground truth and list of other metrics
#The different indices correspond to [DCG, Top-1 Acc, Top-5 Acc, Top-10 Acc, Top-20 Acc, MRR]
def eval_metrics(feature, label, seeddata, glosses, g_dict):
    sortedArgs = minkowski_distance(feature, seeddata, glosses, g_dict)
    pred = sortedArgs[0]
    res, = np.where(sortedArgs == label)
    dcg = 1 / math.log2(res[0] + 1 + 1) #res values start from 0
    mrr = 1 / (res[0] + 1) 
    if res < 1:
        return pred, res[0], [dcg, 1, 1, 1, 1, mrr]
    elif res < 5:
        return pred, res[0], [dcg, 0, 1, 1, 1, mrr]
    elif res < 10:
        return pred, res[0], [dcg, 0, 0, 1, 1, mrr]
    elif res < 20:
        return pred, res[0], [dcg, 0, 0, 0, 1, mrr]
    else:
        return pred, res[0], [dcg, 0, 0, 0, 0, mrr]


seeddata, glosses, g_dict = load_data(seed_file)
testdata, labels, _ = load_data(test_file)

count_total = 0
#the different indices correspond to sum [DCG, Top-1, Top-5, Top-10, Top-20, MRR]
count_correct = [0, 0, 0, 0, 0, 0]
conf_matrix = np.zeros((len(glosses), len(glosses)))
gloss_count = np.zeros(len(glosses))

start = timer()
for i in range(len(testdata)):
    #for timer
    if i % 500 == 0:
        end =  timer()
        print(str(i) + " " + str(end - start))
        start = end

    features = testdata[i]
    g = labels[i]
    label = g_dict[g]
 
    pred, res, counts = eval_metrics(features, label, seeddata, glosses, g_dict)
    count_correct = list(map(add, counts, count_correct))
    count_total = count_total + 1
    conf_matrix[label, pred] =  conf_matrix[label, pred] + 1
    gloss_count[label] = gloss_count[label] + 1

with open('output ' + output_name + '.txt', 'w') as f:
    f.write("Total files in eval = " + str(count_total) + '\n')
    f.write("Cosine distance"+ '\n')
    f.write("Discounted Cumulative Gain is " + str(count_correct[0]/count_total)+ '\n')
    f.write("Mean Reciprocal Rank is " + str(count_correct[5]/count_total)+ '\n')
    f.write("Top-1 accuracy is " + str(count_correct[1]/count_total)+ '\n')
    f.write("Top-5 accuracy is " + str(count_correct[2]/count_total)+ '\n')
    f.write("Top-10 accuracy is " + str(count_correct[3]/count_total)+ '\n')
    f.write("Top-20 accuracy is " + str(count_correct[4]/count_total)+ '\n')
    f.write('\n')

np.savetxt('confusion matrix ' + output_name + '.txt', conf_matrix, fmt='%d')
