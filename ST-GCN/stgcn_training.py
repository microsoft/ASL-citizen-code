import math
import os
import argparse
import random

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

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_default_dtype(torch.float64)
device = 'cuda:3'

np.set_printoptions(threshold=10_000)

#Update files and paths as needed
video_base_path = '../data/PaidDataSet/poses/'
train_file = '../data_csv/aslcitizen_training_set.csv'
test_file = '../data_csv/aslcitizen_validation_set.csv'

#Update names according to experiment number
save_model = "./saved_weights_jan_1a/"
dataset_name = "training_full"
logfolder = 'logs_jan_1a'

if not os.path.exists(save_model):
    os.makedirs(save_model)
if not os.path.exists(logfolder):
    os.mkdir(logfolder)

def seed_worker(worker_id):
    #worker_seed = torch.initial_seed() % 2**32
    np.random.seed(0)
    random.seed(0)

g = torch.Generator()
g.manual_seed(0)
train_transforms = pose_transforms.Compose([pose_transforms.ShearTransform(0.1),
                                            pose_transforms.RotatationTransform(0.1)])

#Load datasets
train_ds = Dataset(datadir=video_base_path, video_file=train_file, transforms=train_transforms, pose_map_file="pose_mapping_train.csv")
test_ds = Dataset(datadir=video_base_path, video_file=test_file, gloss_dict=train_ds.gloss_dict, pose_map_file="pose_mapping_val.csv")
n_classes = len(train_ds.gloss_dict)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=3, pin_memory=True, worker_init_fn=seed_worker, generator=g)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=1, pin_memory=True,
                                          drop_last=False, worker_init_fn=seed_worker, generator=g)

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
pose_model.cuda()

lr = 1e-3
optimizer = optim.Adam(pose_model.parameters(), lr=lr)
max_epoch = 100

num_steps_per_update = 1
steps = 0
epoch = 0
best_val_score = 0
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, last_epoch=-1, T_max=10)

ce_loss = torch.nn.CrossEntropyLoss()

while epoch < max_epoch:
    #setup for writing logs
    output = open(logfolder + "/log" + str(epoch) + ".txt", 'w')
    output.write('Step {}/{}\n'.format(steps, 64000))
    output.write('-' * 10)
    output.write('\n')

    i = 0
    epoch += 1

    # Each epoch has a training and validation phase
    for phase in ['train', 'test']:
        if phase == 'train':
            pose_model.train(True)
        else:
            pose_model.train(False)  # Set model to evaluate mode

        tot_loss = 0.0 
        tot_cls_loss = 0.0
        num_iter = 0
        correct = 0
        total = 0
        optimizer.zero_grad()

        # Iterate over data.
        curr_dataloader = train_loader
        if phase == 'test':
            curr_dataloader = test_loader
            i = 0
        
        for data in tqdm(curr_dataloader):
            num_iter += 1

            inputs, name, labels = data

            # wrap them in Variable
            inputs = inputs.cuda()
            labels = labels.cuda()

            #pass through model, calculate loss
            outputs = pose_model(inputs)
            cls_loss = ce_loss(outputs, labels)
            tot_cls_loss += cls_loss.data.item()

            #calculate number correct
            y_pred_tag = torch.softmax(outputs, dim=1)
            pred_args = torch.argmax(y_pred_tag, dim=1)
            true_args = torch.argmax(labels, dim=1)
            correct += (pred_args == true_args).sum().float()
            total += labels.shape[0]

            loss = cls_loss / num_steps_per_update
            tot_loss += loss.data.item()

            if num_iter == num_steps_per_update // 2:
                output.write(epoch, steps, loss.data.item())
                output.write('\n')
            if phase == 'train':
                loss.backward()

            if num_iter == num_steps_per_update and phase == 'train':
                steps += 1
                num_iter = 0
                optimizer.step()
                optimizer.zero_grad()
                #update logs
                if steps % 10 == 0:
                    acc = float(correct / total)
                    output.write(
                        'Epoch {} {} Cls Loss: {:.4f} Tot Loss: {:.4f} Accu :{:.4f}'.format(epoch,
                                                                                             phase,
                                                                                             tot_cls_loss / (10 * num_steps_per_update),
                                                                                             tot_loss / 10,
                                                                                             acc))
                    output.write('\n')
                    print((
                        'Epoch {} {} Cls Loss: {:.4f} Tot Loss: {:.4f} Accu :{:.4f}'.format(epoch,
                                                                                             phase,
                                                                                             tot_cls_loss / (10 * num_steps_per_update),
                                                                                             tot_loss / 10,
                                                                                             acc)))
                    tot_loss = tot_cls_loss = 0.

        if phase == 'test':
            val_score = float(correct / total)

            #save best model or on even epochs
            if val_score > best_val_score or epoch % 2 == 0:
               
                model_name = save_model + "_" + dataset_name + str(steps).zfill(6) + '_%3f.pt' % val_score
                torch.save(pose_model.state_dict(), model_name)
                output.write(model_name)
                output.write('\n')

            output.write('VALIDATION: {}  Cls Loss: {:.4f} Tot Loss: {:.4f} Accu :{:.4f}'.format(phase,
                                                                                                  tot_cls_loss / num_iter,
                                                                                                  (tot_loss * num_steps_per_update) / num_iter,
                                                                                                  val_score
                                                                                                  ))
            output.write('\n')
            print('VALIDATION: {}  Cls Loss: {:.4f} Tot Loss: {:.4f} Accu :{:.4f}'.format(phase,
                                                                                                  tot_cls_loss / num_iter,
                                                                                                  (tot_loss * num_steps_per_update) / num_iter,
                                                                                                  val_score
                                                                                                  ))
            scheduler.step(tot_loss * num_steps_per_update / num_iter)
            if val_score > best_val_score:
                best_val_score = val_score


    output.close()
