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

from asl_citizen_dataset import ASLCitizen as Dataset
import cv2
from tqdm import tqdm
from operator import add

#import seaborn as sns

from timeit import default_timer as timer

parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('seed_file', type=str,
                    help='Seed File')
parser.add_argument('test_file', type=str,
                    help='Test File')
parser.add_argument('output_name', type=str,
                    help='Output file name')
parser.add_argument('--normalize', action=argparse.BooleanOptionalAction)

args = parser.parse_args()
seed_file = args.seed_file
test_file = args.test_file
output_name = args.output_name
norm = args.normalize

def load_seed(file):
    print(file)
    with open(file) as f:
        ncols = len(f.readline().split('\t'))
    seeddata = np.loadtxt(file, delimiter='\t', usecols=range(1,ncols))
    with open(file) as f:
        reader = csv.reader(f, delimiter="\t")
        glosses = []
        for row in reader:
            gloss = row[0].strip()
            glosses.append(gloss) 

    return seeddata, glosses

def load_test(file):
    print(file)
    with open(file) as f:
        ncols = len(f.readline().split('\t'))
    testdata = np.loadtxt(file, delimiter='\t', usecols=range(1,ncols))
    with open(file) as f:
        reader = csv.reader(f, delimiter="\t")
        labels = []
        for row in reader:
            label = row[0].strip()
            labels.append(label) 

    return testdata, labels

def minkowski_distance(feature, seeddata, glosses, g_dict):
    distances = [0] * len(g_dict)
    gloss_dist = {}
    for i in range(len(seeddata)):
        train_gloss = glosses[i]
        if train_gloss in g_dict:
            train_feat = seeddata[i]
            #train_gloss = glosses[i]
            dist = cosine(train_feat, feature)

            if train_gloss not in gloss_dist:
                gloss_dist[train_gloss] = dist
            elif dist < gloss_dist[train_gloss]:
                gloss_dist[train_gloss] = dist

    for g in gloss_dist:
        index = g_dict[g]
        distances[index] = gloss_dist[g]

#    for f in seeddata:
#        dist = cosine(f, feature)
#        distances.append(dist)
    sortedargs = np.argsort(distances)
    return sortedargs

def eval_metrics(feature, label, seeddata, glosses, g_dict):
    sortedArgs = minkowski_distance(feature, seeddata, glosses, g_dict)
    pred = sortedArgs[0]
    #res = sortedArgs.index(label)
    res, = np.where(sortedArgs == label)
    #print(res[0])
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


#seeddata, glosses = load_seed('seed_data_supervised.tsv')
seeddata, glosses = load_seed(seed_file)

if norm:
    mean = np.loadtxt('mean-seed.txt')
    sd = np.loadtxt('sd-seed.txt')

def normalize(feature):
    diff = np.subtract(feature, mean)
    norm = np.divide(diff, sd)
    return norm

asl_subset = []
with open('../utils/overlapping_asl.txt') as f:
    for line in f:
        line = line.strip()
        asl_subset.append(line)

g_index = 0
g_dict = {}
for i in range(len(glosses)):
    g = glosses[i]
    if g not in g_dict and g in asl_subset:
        g_dict[g] = g_index
        g_index += 1

count_total = 0
count_correct = [0, 0, 0, 0, 0, 0]
conf_matrix = np.zeros((len(glosses), len(glosses)))
gloss_count = np.zeros(len(glosses))
user_stats = {}
user_counts = {}

#testdata, labels = load_test('test_data_supervised.tsv')
testdata, labels = load_test(test_file)

if norm:
    for i in range(len(seeddata)):
       seeddata[i] = normalize(seeddata[i])

    for i in range(len(testdata)):
       testdata[i] = normalize(testdata[i])
start = timer()
for i in range(len(testdata)):
    if i % 500 == 0:
        end =  timer()
        print(str(i) + " " + str(end - start))
        start = end
    features = testdata[i]
    g = labels[i]
    #u = name['user'][0]

    if g in g_dict:
        label = g_dict[g]
    else:
        label = -1

    if label != -1: 
        pred, res, counts = eval_metrics(features, label, seeddata, glosses, g_dict)
        count_correct = list(map(add, counts, count_correct))
        count_total = count_total + 1
        conf_matrix[label, pred] =  conf_matrix[label, pred] + 1
        gloss_count[label] = gloss_count[label] + 1
        '''
        if u not in user_counts:
            user_counts[u] = 1
            user_stats[u] = counts
        else:
            user_counts[u] = user_counts[u] + 1
            user_stats[u] = list(map(add, counts, user_stats[u]))
        '''

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

'''
with open('user_stats.txt', 'w') as f:
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
'''
np.savetxt('confusion matrix ' + output_name + '.txt', conf_matrix, fmt='%d')
