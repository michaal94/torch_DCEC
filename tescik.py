import torch.nn as nn
from torchvision import datasets, models, transforms
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import DCEC
import time
import copy
import matplotlib.pyplot as plt

batch=100

data_dir = 'data'
img_size = [128,128]
# Transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Read data from selected folder and apply transformations
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
# Prepare data for network: schuffle and arrange batches
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                              shuffle=True, num_workers=4)
               for x in ['train', 'val']}

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=1, shuffle=True, num_workers=4)

dataloaders['train']=train_loader
dataloaders['val']=test_loader
dataset_sizes={'train': 60000}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
#     # Note the time
#     since = time.time()
#
#     # Prep variables for weights and accuracy of the best model
#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_acc = 0.0
#     best_loss = 10000.0
#
#     # Go through all epochs
#     for epoch in range(num_epochs):
#         print('Epoch {}/{}'.format(epoch + 1, num_epochs))
#         print('-' * 10)
#
#         for phase in ['train']:
#             print("Running {} phase".format('training' if phase == 'train' else 'validation'))
#             if phase == 'train':
#                 scheduler.step()
#                 model.train(True)  # Set model to training mode
#             else:
#                 model.train(False)  # Set model to evaluate mode
#
#             running_loss = 0.0
#             running_corrects = 0
#
#             # Keep the batch number for inter-phase statistics
#             batch_num = 1
#
#             # Iterate over data.
#             for i, data in enumerate(dataloaders[phase]):
#                 print(i)
#
#                 # Get the inputs and labels
#                 inputs, labels = data
#
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)
#
#                 # zero the parameter gradients
#                 optimizer.zero_grad()
#
#                 with torch.set_grad_enabled(phase == 'train'):
#                     outputs = model(inputs)
#                     loss = criterion(outputs, inputs)
#                     # loss = Variable(loss, requires_grad=True)
#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()
#
#                 # For keeping statistics
#                 running_loss += loss.item() * inputs.size(0)
#                 # running_corrects += torch.sum(preds == labels.data)
#
#                 # Some current stats
#                 loss_batch = loss.item()
#                 loss_accum = running_loss / ((batch_num - 1) * batch + inputs.size(0))
#                 # acc_batch = torch.sum(preds == labels.data).double() / inputs.size(0)
#                 # acc_accum = running_corrects.double() / ((batch_num - 1) * batch + inputs.size(0))
#
#                 if batch_num % 1 == 0:
#                     print('Epoch: [{0}][{1}/{2}]\t'
#                                'Loss {3:.4f} ({4:.4f})\t'.format(epoch + 1, batch_num, len(dataloaders[phase]),
#                                                                  loss_batch,
#                                                                  loss_accum))
#                     # if board:
#                     #     niter = epoch * len(dataloaders[phase]) + batch_num
#                     #     writer.add_scalar(phase.title() + '/Loss', loss_accum, niter)
#                     #     writer.add_scalar(phase.title() + '/Acc', acc_accum, niter)
#
#                 batch_num = batch_num + 1
#
#                 if i == len(dataloaders[phase])-1:
#                     fig = plt.figure()
#                     ax = fig.add_subplot(1,2,1)
#                     inp = inputs.cpu().data[0].numpy().transpose((1, 2, 0))
#                     mean = np.array([0.485, 0.456, 0.406])
#                     std = np.array([0.229, 0.224, 0.225])
#                     inp = std * inp + mean
#                     inp = np.clip(inp, 0, 1)
#                     ax.imshow(inp)
#                     ax = fig.add_subplot(1, 2, 2)
#                     inp = outputs.cpu().data[0].numpy().transpose((1, 2, 0))
#                     mean = np.array([0.485, 0.456, 0.406])
#                     std = np.array([0.229, 0.224, 0.225])
#                     inp = std * inp + mean
#                     inp = np.clip(inp, 0, 1)
#                     ax.imshow(inp)
#                 plt.show()
#
#             epoch_loss = running_loss / dataset_sizes[phase]
#             # epoch_acc = running_corrects.double() / dataset_sizes[phase]
#
#             # if board:
#             #     writer.add_scalar(phase.title() + '/Loss' + '/Epoch', epoch_loss, epoch + 1)
#             #     writer.add_scalar(phase.title() + '/Acc' + '/Epoch', epoch_acc, epoch + 1)
#
#             print('{}\t Loss: {:.4f}'.format(
#                 phase.title(), epoch_loss))
#             best_model_wts = copy.deepcopy(model.state_dict())
#
#             # # deep copy the
#             # if phase == 'val' and epoch_acc > best_acc:
#             #     # if phase == 'val' and (epoch_acc > best_acc or (epoch_acc == best_acc and epoch_loss < best_loss)):
#             #     best_acc = epoch_acc
#             #     best_loss = epoch_loss
#             #     best_model_wts = copy.deepcopy(model.state_dict())
#
#         # print_both('')
#
#     time_elapsed = time.time() - since
#     print('Training complete in {:.0f}m {:.0f}s'.format(
#         time_elapsed // 60, time_elapsed % 60))
#     # print_both('Best val accuracy: {:4f}'.format(best_acc))
#
#     # load best model weights
#     model.load_state_dict(best_model_wts)
#     return model


model = DCEC.DCEC([28,28,1])
model = model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
# optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01, momentum=0.9,
#                          weight_decay=0)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)


# model = train_model(model, criterion, optimizer, exp_lr_scheduler,
#                        num_epochs=200)

for i, data in enumerate(dataloaders['train']):
    img, _ = data
    img = img.to(device)
    out, cl_out = model(img)
    print(out.size())
    print(cl_out.size())
    if i == 0: break

print(model)

model.eval()

