from __future__ import print_function, division

if __name__ == "__main__":

    import argparse
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim import lr_scheduler
    from torchvision import datasets, models, transforms
    import os
    import math
    import fnmatch
    import nets
    import utils
    import training_functions
    import distutils
    from tensorboardX import SummaryWriter


    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description='Use DCEC for clustering')
    parser.add_argument('--mode', default='train_full', choices=['train_full', 'pretrain'], help='mode')
    parser.add_argument('--tensorboard', default=True, type=bool, help='export training stats to tensorboard')
    parser.add_argument('--pretrain', default=True, type=str2bool, help='perform autoencoder pretraining')
    parser.add_argument('--pretrained_net', default=1, help='index or path of pretrained net')
    parser.add_argument('--net_architecture', default='CAE_3', choices=['CAE_3', 'CAE_bn3', 'CAE_4', 'CAE_bn4', 'CAE_5', 'CAE_bn5'], help='network architecture used')
    parser.add_argument('--dataset', default='MNIST', choices=['MNIST', 'custom', 'MNIST-test'],
                        help='custom or prepared dataset')
    parser.add_argument('--dataset_path', default='data', help='path to dataset')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--rate', default=0.001, type=float, help='learning rate for clustering')
    parser.add_argument('--rate_pretrain', default=0.001, type=float, help='learning rate for pretraining')
    parser.add_argument('--weight', default=0.0, type=float, help='weight decay for clustering')
    parser.add_argument('--weight_pretrain', default=0.0, type=float, help='weight decay for clustering')
    parser.add_argument('--sched_step', default=200, type=int, help='scheduler steps for rate update')
    parser.add_argument('--sched_step_pretrain', default=200, type=int,
                        help='scheduler steps for rate update - pretrain')
    parser.add_argument('--sched_gamma', default=0.1, type=float, help='scheduler gamma for rate update')
    parser.add_argument('--sched_gamma_pretrain', default=0.1, type=float,
                        help='scheduler gamma for rate update - pretrain')
    parser.add_argument('--epochs', default=1000, type=int, help='clustering epochs')
    parser.add_argument('--epochs_pretrain', default=300, type=int, help='pretraining epochs')
    parser.add_argument('--printing_frequency', default=10, type=int, help='training stats printing frequency')
    parser.add_argument('--gamma', default=0.1, type=float, help='clustering loss weight')
    parser.add_argument('--gamma_lab', default=0.01, type=float, help='labelled loss weight')
    parser.add_argument('--update_interval', default=80, type=int, help='update interval for target distribution')
    parser.add_argument('--label_upd_interval', default=1, type=int, help='update interval for target distribution')
    parser.add_argument('--tol', default=1e-2, type=float, help='stop criterium tolerance')
    parser.add_argument('--num_clusters', default=10, type=int, help='number of clusters')
    parser.add_argument('--custom_img_size', default=[128, 128, 3], nargs=3, type=int, help='size of custom images')
    parser.add_argument('--leaky', default=True, type=str2bool)
    parser.add_argument('--neg_slope', default=0.01, type=float)
    parser.add_argument('--activations', default=False, type=str2bool)
    parser.add_argument('--bias', default=True, type=str2bool)
    args = parser.parse_args()
    print(args)

    if args.mode == 'pretrain' and not args.pretrain:
        print("Nothing to do :(")
        exit()

    board = args.tensorboard

    pretrain = args.pretrain
    net_is_path = True
    if not pretrain:
        try:
            int(args.pretrained_net)
            idx = args.pretrained_net
            net_is_path = False
        except:
            pass
    params = {'pretrain': pretrain}

    # Directories
    dirs = ['runs', 'reports', 'nets']
    list(map(lambda x: os.makedirs(x, exist_ok=True), dirs))

    # Net architecture
    model_name = args.net_architecture
    # Indexing
    if pretrain or (not pretrain and net_is_path):
        reports_list = sorted(os.listdir('reports'), reverse=True)
        if reports_list:
            for file in reports_list:
                # print(file)
                if fnmatch.fnmatch(file, model_name+'*'):
                    print(file)
                    idx = int(str(file)[-7:-4]) + 1
                    print(idx)
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

    print(name_txt)

    name_txt = os.path.join('reports', name_txt)
    name_net = os.path.join('nets', name_net)
    if net_is_path and not pretrain:
        pretrained = args.pretrained_net
    else:
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

    # Used dataset
    dataset = args.dataset

    # Batch size
    batch = args.batch_size
    params['batch'] = batch
    # Number of workers (typically 4*num_of_GPUs)
    workers = 4
    # Learning rate
    rate = args.rate
    rate_pretrain = args.rate_pretrain
    # Adam params
    # Weight decay
    weight = args.weight
    weight_pretrain = args.weight_pretrain
    # Scheduler steps for rate update
    sched_step = args.sched_step
    sched_step_pretrain = args.sched_step_pretrain
    # Scheduler gamma - multiplier for learning rate
    sched_gamma = args.sched_gamma
    sched_gamma_pretrain = args.sched_gamma_pretrain

    # Number of epochs
    epochs = args.epochs
    pretrain_epochs = args.epochs_pretrain
    params['pretrain_epochs'] = pretrain_epochs

    # Printing frequency
    print_freq = args.printing_frequency
    params['print_freq'] = print_freq

    # Clustering loss weight:
    gamma = args.gamma
    params['gamma'] = gamma

    # Labelled loss weight:
    gamma_lab = args.gamma_lab
    params['gamma_lab'] = gamma_lab

    # Update interval for target distribution:
    update_interval = args.update_interval
    params['update_interval'] = update_interval

    label_upd_interval = args.label_upd_interval
    params['label_upd_interval'] = label_upd_interval

    # Tolerance for label changes:
    tol = args.tol
    params['tol'] = tol

    # Number of clusters
    num_clusters = args.num_clusters

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
    tmp = "Pretraining learning rate:\t" + str(rate_pretrain)
    utils.print_both(f, tmp)
    tmp = "Weight decay:\t" + str(weight)
    utils.print_both(f, tmp)
    tmp = "Pretraining weight decay:\t" + str(weight_pretrain)
    utils.print_both(f, tmp)
    tmp = "Scheduler steps:\t" + str(sched_step)
    utils.print_both(f, tmp)
    tmp = "Scheduler gamma:\t" + str(sched_gamma)
    utils.print_both(f, tmp)
    tmp = "Pretraining scheduler steps:\t" + str(sched_step_pretrain)
    utils.print_both(f, tmp)
    tmp = "Pretraining scheduler gamma:\t" + str(sched_gamma_pretrain)
    utils.print_both(f, tmp)
    tmp = "Number of epochs of training:\t" + str(epochs)
    utils.print_both(f, tmp)
    tmp = "Number of epochs of pretraining:\t" + str(pretrain_epochs)
    utils.print_both(f, tmp)
    tmp = "Clustering loss weight:\t" + str(gamma)
    utils.print_both(f, tmp)
    tmp = "Labelled loss weight:\t" + str(gamma_lab)
    utils.print_both(f, tmp)
    tmp = "Update interval for target distribution:\t" + str(update_interval)
    utils.print_both(f, tmp)
    tmp = "Update interval for labelled loss:\t" + str(label_upd_interval)
    utils.print_both(f, tmp)
    tmp = "Stop criterium tolerance:\t" + str(tol)
    utils.print_both(f, tmp)
    tmp = "Number of clusters:\t" + str(num_clusters)
    utils.print_both(f, tmp)
    tmp = "Leaky relu:\t" + str(args.leaky)
    utils.print_both(f, tmp)
    tmp = "Leaky slope:\t" + str(args.neg_slope)
    utils.print_both(f, tmp)
    tmp = "Activations:\t" + str(args.activations)
    utils.print_both(f, tmp)
    tmp = "Bias:\t" + str(args.bias)
    utils.print_both(f, tmp)

    # Data preparation
    if dataset == 'MNIST':
        tmp = "\nData preparation\nReading data from: MNIST dataset"
        utils.print_both(f, tmp)
        img_size = [28, 28, 1]
        tmp = "Image size used:\t{0}x{1}".format(img_size[0], img_size[1])
        utils.print_both(f, tmp)

        dataset = datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               # transforms.Normalize((0.1307,), (0.3081,))
                           ]))

        dataloader = torch.utils.data.DataLoader(dataset,
            batch_size=batch, shuffle=False, num_workers=workers)

        dataset_size = 60000
        tmp = "Training set size:\t" + str(dataset_size)
        utils.print_both(f, tmp)

        dataset_labelled = datasets.MNIST('../data', train=False, download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     # transforms.Normalize((0.1307,), (0.3081,))
                                 ]))

        dataloader_labelled = torch.utils.data.DataLoader(dataset_labelled,
                                                 batch_size=batch, shuffle=False, num_workers=workers)

        dataset_labelled_size = 10000


    elif dataset == 'MNIST-test':
        tmp = "\nData preparation\nReading data from: MNIST test dataset"
        utils.print_both(f, tmp)
        img_size = [28, 28, 1]
        tmp = "Image size used:\t{0}x{1}".format(img_size[0], img_size[1])
        utils.print_both(f, tmp)

        dataloader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               # transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=batch, shuffle=False, num_workers=workers)

        dataset_size = 10000
        tmp = "Training set size:\t" + str(dataset_size)
        utils.print_both(f, tmp)
    else:
        # Data folder
        data_dir = args.dataset_path
        tmp = "\nData preparation\nReading data from:\t./" + data_dir
        utils.print_both(f, tmp)

        # Image size
        custom_size = math.nan
        custom_size = args.custom_img_size
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
                                                      shuffle=False, num_workers=workers)

        # Size of data sets
        dataset_size = len(image_dataset)
        tmp = "Training set size:\t" + str(dataset_size)
        utils.print_both(f, tmp)

    params['dataset_size'] = dataset_size
    params['dataset_labelled_size'] = dataset_labelled_size

    # GPU check
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tmp = "\nPerforming calculations on:\t" + str(device)
    utils.print_both(f, tmp + '\n')
    params['device'] = device

    # print(params)

    to_eval = "nets." + model_name + "(img_size, num_clusters=num_clusters, leaky = args.leaky, neg_slope = args.neg_slope)"
    # model = nets.CAE_3(img_size, num_clusters=num_clusters)
    model = eval(to_eval)

    # Tensorboard model representation
    # if board:
    #     writer.add_graph(model, torch.autograd.Variable(torch.Tensor(batch, img_size[2], img_size[0], img_size[1])))

    model = model.to(device)
    criterion_1 = nn.MSELoss(size_average=True)
    criterion_2 = nn.KLDivLoss(size_average=False)
    criterion_3 = nn.CrossEntropyLoss(size_average=False)

    criteria = [criterion_1, criterion_2, criterion_3]

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=rate, weight_decay=weight)

    optimizer_pretrain = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=rate_pretrain, weight_decay=weight_pretrain)

    optimizers = [optimizer, optimizer_pretrain]

    scheduler = lr_scheduler.StepLR(optimizer, step_size=sched_step, gamma=sched_gamma)
    scheduler_pretrain = lr_scheduler.StepLR(optimizer_pretrain, step_size=sched_step_pretrain, gamma=sched_gamma_pretrain)

    schedulers = [scheduler, scheduler_pretrain]

    print([dataloader, dataloader_labelled])

    if args.mode == 'train_full':
        model = training_functions.train_semisupervised(model, [dataloader, dataloader_labelled], criteria, optimizers, schedulers, epochs, params)
    elif args.mode == 'pretrain':
        model = training_functions.pretraining(model, [dataloader, dataloader_labelled], criteria, optimizers, schedulers, epochs, params)

    torch.save(model.state_dict(), name_net + '.pt')

    f.close()
    writer.close()

