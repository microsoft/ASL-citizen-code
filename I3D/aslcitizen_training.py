import math
import os
import argparse
import random

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

random.seed(0)
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
test_file = '../../final_dataset/ASL_Citizen/splits/val.csv'

#Update names according to experiment number
save_model = "./saved_weights_may/"
dataset_name = "v1"
log_file = 'logs_may'

if not os.path.exists(save_model):
    os.makedirs(save_model)

if not os.path.exists(log_file):
    os.mkdir(log_file)

def seed_worker(worker_id):
    #worker_seed = torch.initial_seed() % 2**32
    np.random.seed(0)
    random.seed(0)
    

g = torch.Generator()
g.manual_seed(0)

#Load datasets
train_ds = Dataset(datadir=video_base_path, transforms=train_transforms, video_file=train_file)
print(len(train_ds.gloss_dict))

test_ds = Dataset(datadir=video_base_path, transforms=test_transforms, video_file=test_file, gloss_dict=train_ds.gloss_dict)
print(len(test_ds.gloss_dict))

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=True, worker_init_fn=seed_worker, generator=g)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=True, worker_init_fn=seed_worker, generator=g)

# Load i3d model
i3d = InceptionI3d(400, in_channels=3)
i3d.replace_logits(2731)
i3d.cuda()

lr = 1e-3
weight_decay = 1e-8
optimizer = optim.Adam(i3d.parameters(), lr=lr, weight_decay=weight_decay)

num_steps_per_update = 1
steps = 0
epoch = 0
best_val_score = 0
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.3, verbose=True)

while epoch < 75:
    #setup for writing logs
    output = open(log_file + "/log" + str(epoch) + ".txt", 'w')
    output.write('Step {}/{}\n'.format(steps, 64000))
    output.write('-' * 10)
    output.write('\n')

    epoch += 1
    # Each epoch has a training and validation phase
    for phase in ['train', 'test']:
        if phase == 'train':
            i3d.train(True)
        else:
            i3d.train(False)  # Set model to evaluate mode

        tot_loss = 0.0 # mix of localization and classification loss
        tot_loc_loss = 0.0
        tot_cls_loss = 0.0
        num_iter = 0
        correct = 0 #num correctly classified videos
        total = 0 #num total videos
        optimizer.zero_grad()

        # Iterate over data.
        curr_dataloader = train_loader
        if phase == 'test':
            curr_dataloader = test_loader
        for data in tqdm(curr_dataloader):
            num_iter += 1
            
            inputs, name, labels = data

            # wrap them in Variable
            inputs = inputs.cuda()
            t = inputs.size(2)
            labels = labels.cuda()

            per_frame_logits = i3d(inputs, pretrained=False)
            ground_truth = torch.max(labels, dim=2)[0]

            # upsample to input size
            per_frame_logits = F.upsample(per_frame_logits, t, mode='linear')

            # compute localization loss
            loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
            tot_loc_loss += loc_loss.data.item()
            predictions = torch.max(per_frame_logits, dim=2)[0]

            # compute classification loss (with max-pooling along time B x C x T)
            cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0], ground_truth)
            tot_cls_loss += cls_loss.data.item()

            #calculate correctly classified videos
            y_pred_tag = torch.softmax(predictions, dim=1)
            pred_args = torch.argmax(y_pred_tag, dim=1)
            true_args = torch.argmax(ground_truth, dim=1)
            correct += (pred_args == true_args).sum().float()
            total += ground_truth.shape[0]

            loss = (0.5 * loc_loss + 0.5 * cls_loss) / num_steps_per_update
            tot_loss += loss.data.item()
            if num_iter == num_steps_per_update // 2:
                output.write(epoch, steps, loss.data.item())
                output.write('\n')
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
                        'Epoch {} {} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f} Accu :{:.4f}'.format(epoch,
                                                                                                             phase,
                                                                                                             tot_loc_loss / (
                                                                                                                         10 * num_steps_per_update),
                                                                                                             tot_cls_loss / (
                                                                                                                         10 * num_steps_per_update),
                                                                                                             tot_loss / 10,
                                                                                                             acc))
                    output.write('\n')
                    tot_loss = tot_loc_loss = tot_cls_loss = 0.

        if phase == 'test':
            val_score = float(correct / total)
            #save best model or on even epochs
            if val_score > best_val_score or epoch % 2 == 0:
                
                model_name = save_model + "_" + dataset_name + str(epoch) + '_%3f.pt' % val_score
                torch.save(i3d.state_dict(), model_name)
                output.write(model_name)
                output.write('\n')

            output.write('VALIDATION: {} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f} Accu :{:.4f}'.format(phase,
                                                                                                          tot_loc_loss / num_iter,
                                                                                                          tot_cls_loss / num_iter,
                                                                                                          (
                                                                                                                      tot_loss * num_steps_per_update) / num_iter,
                                                                                                          val_score
                                                                                                          ))
            output.write('\n')
            scheduler.step(tot_loss * num_steps_per_update / num_iter)
            if val_score > best_val_score:
                best_val_score = val_score


    output.close()
