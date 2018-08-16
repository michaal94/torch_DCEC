import torch.nn as nn
from torchvision import datasets, models, transforms
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import nets
import time
import copy
import matplotlib.pyplot as plt

batch=100

data_dir = 'data'
img_size = [128,128, 3]
# Transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(img_size[0:2]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(img_size[0:2]),
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

# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=True, download=True,
#                    transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ])),
#     batch_size=batch, shuffle=True, num_workers=4)
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=False, transform=transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,))
#     ])),
#     batch_size=1, shuffle=True, num_workers=4)
#
# dataloaders['train']=train_loader
# dataloaders['val']=test_loader
# dataset_sizes={'train': 60000}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model = nets.CAE_bn5(img_size, leaky=True, neg_slope=0.01, activations=True, bias=True)
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
    out, cl_out, _ = model(img)
    print(out.size())
    print(cl_out.size())
    if i == 0: break

# print(model)

model.eval()

