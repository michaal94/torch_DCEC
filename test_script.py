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

from tensorboardX import SummaryWriter

board = True
test = True

pretrain = False
idx = 1

# Directories
dirs = ['runs', 'reports', 'nets']
map(lambda x: os.makedirs(x, exist_ok=True), dirs)

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
name_net = name + '.pt'
pretrained = name + '_pretrained.pt'

name_txt = os.path.join('reports', name_txt)
name_net = os.path.join('nets', name_net)
pretrained = os.path.join('nets', pretrained)
if not pretrain and not os.path.isfile(pretrained):
    print("No pretrained weights, try again choosing pretrained network or create new with pretrain=True")


# Open file
if pretrain:
    f = open(name_txt, 'w')
else:
    f = open(name_txt, 'a')


# Define printing to console and file
def print_both(text):
    print(text)
    f.write(text + '\n')

# Delete tensorboard entry if exist (not to overlap)
try:
    os.system("rm -rf runs/" + name)
except:
    pass

# Initialize tensorboard writer
if board:
    writer = SummaryWriter('runs/' + name)

# Hyperparameters

dataset = 'MNIST'

# Batch size
batch = 600
# Number of workers (typically 4*num_of_GPUs)
workers = 4
# Learning rate
rate = 0.01
# Adam params
# Weight decay
weight = 0
# Scheduler steps for rate update
sched_step = 20
# Scheduler gamma - multiplier for learning rate
sched_gamma = 0.1

# Number of epochs
epochs = 10

# Printing frequency
print_freq = 10

# Report for settings
tmp = "Training the '" + model_name + "' architecture"
print_both(tmp)
tmp = "\n" + "The following parameters are used:"
print_both(tmp)
tmp = "Batch size:\t" + str(batch)
print_both(tmp)
tmp = "Number of workers:\t" + str(workers)
print_both(tmp)
tmp = "Learning rate:\t" + str(rate)
print_both(tmp)
tmp = "Weight decay:\t" + str(weight)
print_both(tmp)
tmp = "Scheduler steps:\t" + str(sched_step)
print_both(tmp)
tmp = "Scheduler gamma:\t" + str(sched_gamma)
print_both(tmp)
tmp = "Number of epochs of training:\t" + str(epochs)
print_both(tmp)

# Data preparation

if dataset == 'MNIST':
    tmp = "\nData preparation\nReading data from: MNIST dataset"
    print_both(tmp)
    img_size = [28, 28, 1]
    tmp = "Image size used:\t{0}x{1}".format(img_size[0], img_size[1])
    print_both(tmp)

    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch, shuffle=True, num_workers=4)

    dataset_size = 60000
    tmp = "Training set size:\t" + str(dataset_size)
    print_both(tmp)

else:
    # Data folder
    data_dir = 'data'
    tmp = "\nData preparation\nReading data from:\t./" + data_dir
    print_both(tmp)

    # Image size
    custom_size = math.nan
    custom_size = [128,128,3]
    if isinstance(custom_size, list):
        img_size = custom_size

    tmp = "Image size used:\t{0}x{1}".format(img_size[0], img_size[1])
    print_both(tmp)

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
    print_both(tmp)


# Class names
# class_names = image_datasets['train'].classes

# GPU check
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tmp = "\nPerforming calculations on:\t" + str(device)
print_both(tmp + '\n')


# Training function
def train_model(model, criterion, optimizer, scheduler, num_epochs=25, pretrain=True):
    # Note the time
    since = time.time()

    if pretrain:
        while True:
            pretrained_model = pretraining(model, criterion, optimizer, exp_lr_scheduler, num_epochs=6)
            if pretrained_model: break
        model = pretrained_model
    else:
        model.load_state_dict(torch.load(pretrained))

    print_both('\nInitialize cluster centers:')
    model = kmeans(model)



    print_both('\nBegin clusters training:')

    # Prep variables for weights and accuracy of the best model
    best_model_wts = copy.deepcopy(model.state_dict())
    # best_acc = 0.0
    best_loss = 10000.0

    # Go through all epochs
    for epoch in range(num_epochs):
        print_both('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print_both('-' * 10)

        scheduler.step()
        model.train(True)  # Set model to training mode

        running_loss = 0.0
        # running_corrects = 0

        # Keep the batch number for inter-phase statistics
        batch_num = 1

        img_counter = 0

        # Iterate over data.
        for data in dataloader:
            # Get the inputs and labels
            inputs, _ = data

            inputs = inputs.to(device)
            # labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs, _ = model(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()

            # For keeping statistics
            running_loss += loss.item() * inputs.size(0)
            # running_corrects += torch.sum(preds == labels.data)

            # Some current stats
            loss_batch = loss.item()
            loss_accum = running_loss / ((batch_num - 1) * batch + inputs.size(0))
            # acc_batch = torch.sum(preds == labels.data).double() / inputs.size(0)
            # acc_accum = running_corrects.double() / ((batch_num - 1) * batch + inputs.size(0))

            if batch_num % print_freq == 0:
                print_both('Epoch: [{0}][{1}/{2}]\t'
                           'Loss {3:.4f} ({4:.4f})\t'.format(epoch + 1, batch_num, len(dataloader),
                                                             loss_batch,
                                                             loss_accum))
                if board:
                    niter = epoch * len(dataloader) + batch_num
                    writer.add_scalar('/Loss', loss_accum, niter)
            batch_num = batch_num + 1

            if batch_num == len(dataloader) and (epoch+1) % 5:
                inp = tensor2img(inputs)
                out = tensor2img(outputs)
                if board:
                    img = np.concatenate((inp, out), axis=1)
                    writer.add_image('Clustering/Epoch_' + str(epoch + 1).zfill(3) + '/Sample_' + str(img_counter).zfill(2), img)
                    img_counter += 1

        epoch_loss = running_loss / dataset_size

        if board:
            writer.add_scalar('/Loss' + '/Epoch', epoch_loss, epoch + 1)
        print_both('Loss: {:.4f}'.format(epoch_loss))

        # deep copy the
        if epoch_loss < best_loss or epoch_loss > best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        print_both('')

    time_elapsed = time.time() - since
    print_both('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # print_both('Best val accuracy: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def pretraining(model, criterion, optimizer, scheduler, num_epochs=200, pretrain=True):
    # Note the time
    since = time.time()

    # Prep variables for weights and accuracy of the best model
    best_model_wts = copy.deepcopy(model.state_dict())
    # best_acc = 0.0
    best_loss = 10000.0

    # Go through all epochs
    for epoch in range(num_epochs):
        print_both('Pretraining:\tEpoch {}/{}'.format(epoch + 1, num_epochs))
        print_both('-' * 10)

        scheduler.step()
        model.train(True)  # Set model to training mode

        running_loss = 0.0

        # Keep the batch number for inter-phase statistics
        batch_num = 1
        # Images to show
        img_counter = 0

        # Iterate over data.
        for data in dataloader:
            # Get the inputs and labels
            inputs, _ = data
            inputs = inputs.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs, _ = model(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()

            # For keeping statistics
            running_loss += loss.item() * inputs.size(0)

            # Some current stats
            loss_batch = loss.item()
            loss_accum = running_loss / ((batch_num - 1) * batch + inputs.size(0))

            if batch_num % print_freq == 0:
                print_both('Pretraining:\tEpoch: [{0}][{1}/{2}]\t'
                           'Loss {3:.4f} ({4:.4f})\t'.format(epoch + 1, batch_num, len(dataloader),
                                                             loss_batch,
                                                             loss_accum))
                if board:
                    niter = epoch * len(dataloader) + batch_num
                    writer.add_scalar('Pretraining/Loss', loss_accum, niter)
            batch_num = batch_num + 1

            if batch_num in [len(dataloader), len(dataloader)//2, len(dataloader)//4, 3*len(dataloader)//4]:
                inp = tensor2img(inputs)
                out = tensor2img(outputs)
                if board:
                    img = np.concatenate((inp, out), axis=1)
                    writer.add_image('Pretraining/Epoch_' + str(epoch + 1).zfill(3) + '/Sample_' + str(img_counter).zfill(2), img)
                    img_counter += 1

        epoch_loss = running_loss / dataset_size
        if epoch == 0: first_loss = epoch_loss
        if epoch == 4 and epoch_loss / first_loss > 0.8:
            print_both("Loss not converging, starting pretraining again")
            return False

        if board:
            writer.add_scalar('Pretraining/Loss' + '/Epoch', epoch_loss, epoch + 1)

        print_both('Pretraining:\t Loss: {:.4f}'.format(epoch_loss))

        # deep copy the
        if epoch_loss < best_loss or epoch_loss > best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        print_both('')

    time_elapsed = time.time() - since
    print_both('Pretraining complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    model.pretrained = True
    torch.save(model.state_dict(), pretrained)

    return model


def tensor2img(tensor):
    img = tensor.cpu().data[0].numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img





model = DCEC.DCEC(img_size)
# if board:
    # writer.add_graph(model, torch.autograd.Variable(torch.Tensor(batch, img_size[2], img_size[0], img_size[1])))

model = model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=rate)
# optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01, momentum=0.9,
#                          weight_decay=0)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=sched_step, gamma=sched_gamma)


model = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=200, pretrain=pretrain)

torch.save(model.state_dict(), name_net)

# if test:
#     acc, preds_list = test_model(model_ft, criterion)
#     tmp = "\nPredictions list:\n" + str(preds_list)
#     print_both(tmp)
#     visualize_model(model_ft, 6)
#     plt.ioff()
#     plt.show()

f.close()
writer.close()
