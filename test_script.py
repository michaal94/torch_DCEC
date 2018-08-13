from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import math
import fnmatch
import re
import DCEC
import utils
import training_functions

from tensorboardX import SummaryWriter

board = True
test = True

pretrain = True
if not pretrain:
    idx = 6

params = {'pretrain': pretrain}

# Directories
dirs = ['runs', 'reports', 'nets']
list(map(lambda x: os.makedirs(x, exist_ok=True), dirs))

# Net architecture
model_name = 'DCEC'
# Indexing
if pretrain:
    reports_list = sorted(os.listdir('reports'), reverse=True)
    if reports_list:
        for file in reports_list:
            print(file)
            if fnmatch.fnmatch(file, model_name+'*'):
                idx = int(str(file)[-7:-4]) + 1
                break
    try:
        idx
    except NameError:
        idx = 1

# Base filename
name = model_name + '_' + str(idx).zfill(3)

# Filenames for report and weights
name_txt = name + '.txt'
name_net = name
pretrained = name + '_pretrained.pt'

name_txt = os.path.join('reports', name_txt)
name_net = os.path.join('nets', name_net)
pretrained = os.path.join('nets', pretrained)
if not pretrain and not os.path.isfile(pretrained):
    print("No pretrained weights, try again choosing pretrained network or create new with pretrain=True")

model_files = [name_net, pretrained]
params['model_files'] = model_files

# Open file
if pretrain:
    f = open(name_txt, 'w')
else:
    f = open(name_txt, 'a')
params['txt_file'] = f

# Delete tensorboard entry if exist (not to overlap)
try:
    os.system("rm -rf runs/" + name)
except:
    pass

# Initialize tensorboard writer
if board:
    writer = SummaryWriter('runs/' + name)
    params['writer'] = writer
else:
    params['writer'] = None

# Hyperparameters

dataset = 'MNIST'

# Batch size
batch = 256
params['batch'] = batch
# Number of workers (typically 4*num_of_GPUs)
workers = 4
# Learning rate
rate = 0.001
rate_pretrain = 0.001
# Adam params
# Weight decay
weight = 0
# Scheduler steps for rate update
sched_step = 500
sched_step_pretrain = 200
# Scheduler gamma - multiplier for learning rate
sched_gamma = 0.1
sched_gamma_pretrain = 0.1

# Number of epochs
epochs = 1000
pretrain_epochs = 300
params['pretrain_epochs'] = pretrain_epochs

# Printing frequency
print_freq = 1
params['print_freq'] = print_freq

# Clustering loss weight:
gamma = 0.1
params['gamma'] = gamma

# Update interval for target distribution:
update_interval = 1
params['update_interval'] = update_interval

# Tolerance for label changes:
tol = 1e-3
params['tol'] = tol

# Report for settings
tmp = "Training the '" + model_name + "' architecture"
utils.print_both(f, tmp)
tmp = "\n" + "The following parameters are used:"
utils.print_both(f, tmp)
tmp = "Batch size:\t" + str(batch)
utils.print_both(f, tmp)
tmp = "Number of workers:\t" + str(workers)
utils.print_both(f, tmp)
tmp = "Learning rate:\t" + str(rate)
utils.print_both(f, tmp)
tmp = "Weight decay:\t" + str(weight)
utils.print_both(f, tmp)
tmp = "Scheduler steps:\t" + str(sched_step)
utils.print_both(f, tmp)
tmp = "Scheduler gamma:\t" + str(sched_gamma)
utils.print_both(f, tmp)
tmp = "Number of epochs of training:\t" + str(epochs)
utils.print_both(f, tmp)
tmp = "Number of epochs of pretraining:\t" + str(pretrain_epochs)
utils.print_both(f, tmp)
tmp = "Clustering loss weight:\t" + str(gamma)
utils.print_both(f, tmp)
tmp = "Update interval for target distribution:\t" + str(update_interval)
utils.print_both(f, tmp)

# Data preparation

if dataset == 'MNIST':
    tmp = "\nData preparation\nReading data from: MNIST dataset"
    utils.print_both(f, tmp)
    img_size = [28, 28, 1]
    tmp = "Image size used:\t{0}x{1}".format(img_size[0], img_size[1])
    utils.print_both(f, tmp)

    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           # transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch, shuffle=False, num_workers=4)

    dataset_size = 60000
    tmp = "Training set size:\t" + str(dataset_size)
    utils.print_both(f, tmp)

else:
    # Data folder
    data_dir = 'data'
    tmp = "\nData preparation\nReading data from:\t./" + data_dir
    utils.print_both(f, tmp)

    # Image size
    custom_size = math.nan
    custom_size = [128,128,3]
    if isinstance(custom_size, list):
        img_size = custom_size

    tmp = "Image size used:\t{0}x{1}".format(img_size[0], img_size[1])
    utils.print_both(f, tmp)

    # Transformations
    data_transforms = transforms.Compose([
            transforms.Resize(img_size[0:2]),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # Read data from selected folder and apply transformations
    image_dataset = datasets.ImageFolder(data_dir, data_transforms)
    # Prepare data for network: schuffle and arrange batches
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch,
                                                  shuffle=True, num_workers=workers)

    # Size of data sets
    dataset_size = len(image_dataset)
    tmp = "Training set size:\t" + str(dataset_size)
    utils.print_both(f, tmp)


params['dataset_size'] = dataset_size


# GPU check
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tmp = "\nPerforming calculations on:\t" + str(device)
utils.print_both(f, tmp + '\n')
params['device'] = device

# print(params)

model = DCEC.DCEC(img_size)
# if board:
    # writer.add_graph(model, torch.autograd.Variable(torch.Tensor(batch, img_size[2], img_size[0], img_size[1])))

model = model.to(device)
criterion_1 = nn.MSELoss(size_average=True)
criterion_2 = nn.KLDivLoss(size_average=False)

criteria = [criterion_1, criterion_2]

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=rate)
# optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=rate, momentum=0.9)

optimizer_pretrain = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=rate_pretrain)
# optimizer_pretrain = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=rate_pretrain, momentum=0.9)
# optimizer_pretrain = optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=rate_pretrain)

optimizers = [optimizer, optimizer_pretrain]

scheduler = lr_scheduler.StepLR(optimizer, step_size=sched_step, gamma=sched_gamma)
scheduler_pretrain = lr_scheduler.StepLR(optimizer_pretrain, step_size=sched_step_pretrain, gamma=sched_gamma_pretrain)

schedulers = [scheduler, scheduler_pretrain]

model = training_functions.train_model(model, dataloader, criteria, optimizers, schedulers, epochs, params)

torch.save(model.state_dict(), name_net + '.pt')

# if test:
#     acc, preds_list = test_model(model_ft, criterion)
#     tmp = "\nPredictions list:\n" + str(preds_list)
#     print_both(tmp)
#     visualize_model(model_ft, 6)
#     plt.ioff()
#     plt.show()

f.close()
writer.close()
