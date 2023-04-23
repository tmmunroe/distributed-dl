import argparse
import time
from collections import namedtuple
from tabulate import tabulate
import pandas as pd

import math

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data.dataloader import DataLoader
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torchvision.models import resnet18

from utils import TrainingArgs, TrainingMetrics, EpochMetrics

def default_training_args():
    return TrainingArgs(
        devices=[0],
        datapath='./data',
        workers=2,
        learning_rate=0.1,
        momentum=0.9,
        optimizer='sgd',
        epochs=1,
        batch_size=32
    )


def train(nn_model, optimizer, loss_func, dataloader, device, print_batch_values=False) -> EpochMetrics:
    train_loss = 0
    correct = 0
    total = 0
    inputs, targets = None, None
    dataloading_time = 0.
    training_time = 0.
    total_time = 0.
    batch = 0

    # set training mode and create our dataloader iterator
    nn_model.train()
    dataloader_iter = iter(dataloader)

    if device.type == 'cuda': torch.cuda.synchronize()

    epoch_start = time.perf_counter()
    while True:
        # DATALOADER BLOCK
        dataloader_start = time.perf_counter()
        try:
            inputs, targets = next(dataloader_iter)
        except StopIteration:
            break
        dataloading_time += (time.perf_counter() - dataloader_start)

        # TRAINING BLOCK
        training_start = time.perf_counter()
        inputs, targets = inputs.to(device), targets.to(device)
        if device.type == 'cuda': torch.cuda.synchronize()

        optimizer.zero_grad()

        # prediction error
        outputs = nn_model(inputs)
        loss = loss_func(outputs, targets)
        
        # backpropagation
        loss.backward()
        optimizer.step()
        
        # logging
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if device.type == 'cuda': torch.cuda.synchronize()
        training_time += (time.perf_counter() - training_start)

        batch += 1
        if print_batch_values and (batch % 100 == 0 or batch == len(dataloader)):
            print(f"Batch {batch} of {len(dataloader)}, Loss: {train_loss/(batch+1):.3f} | Acc: {100.*correct/total:.3f} ({correct}/{total}")

    if device.type == 'cuda': # make sure gpu ops are done before timing
        torch.cuda.synchronize()

    total_time = time.perf_counter() - epoch_start

    return EpochMetrics(dataloading_time, training_time, total_time, train_loss/(batch+1), correct/total)


def simple_training(args: TrainingArgs) -> TrainingMetrics:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define Data transformer
    rgb_means = (0.4914, 0.4822, 0.4465) # mean from assignment
    rgb_var = (0.2023, 0.1994, 0.2010) # variance from assignment
    rgb_std = tuple([math.sqrt(var) for var in rgb_var]) # get standard deviation from variance
    data_transformer = transforms.Compose([
        transforms.RandomCrop(size=(32,32), padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=rgb_means, std=rgb_std)
    ])

    # Train set Dataloader
    start_init = time.perf_counter()
    # training_data = datasets.CIFAR10(root=args.datapath, train=True, download=True, transform=data_transformer)
    training_data = datasets.CIFAR10(root=args.datapath, train=True, download=True, transform=data_transformer)
    dataset_init_elapsed = time.perf_counter() - start_init
    print('Time creating dataset object: ', dataset_init_elapsed)

    train_dataloader = DataLoader(training_data, batch_size=128, shuffle=True, num_workers=args.workers)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Create ResNet=18 Model and training parameters
    nn_model = DataParallel(resnet18(), device_ids=args.devices)
    nn_model.to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(nn_model.parameters(), lr=args.learning_rate,
                          momentum=args.momentum, weight_decay=5e-4)
    
    training_metrics = TrainingMetrics(args)
    training_metrics.dataset_init_time = dataset_init_elapsed

    if device.type == 'cuda': # make sure gpu ops are done before timing
        torch.cuda.synchronize()
    for epoch in range(args.epochs):
        print(f"Epoch: {epoch+1} of {args.epochs}")

        epoch_metrics = train(nn_model, optimizer, loss_func, train_dataloader, device)
        epoch_metrics.epoch = epoch+1
        training_metrics.add_epoch_metrics(epoch_metrics)

    return training_metrics

def batch_size_generator():
    batch_size = 32
    while True:
        yield batch_size
        batch_size *= 4


def increasing_batch_size(devices:list, batch_sizes:list=None):
    print(devices)

    args = default_training_args()
    args.epochs = 2
    args.devices = devices

    if batch_sizes is None:
        batch_sizes = batch_size_generator()

    headers = ['gpus', 'batch_size', 'time', 'speedup']
    epoch_table = []
    for batch_size in batch_sizes:
        print('BatchSize: ', batch_size)

        args.batch_size = batch_size
        training_metrics = simple_training(args)
        if len(training_metrics.epoch_metrics) != 2:
            raise ValueError(f'Expected 2 epochs to be run.. but {len(epoch_metrics)} were run')
        
        epoch_metrics:EpochMetrics = training_metrics.epoch_metrics[-1]
        
        print('Epoch 2 Metrics:')
        print(epoch_metrics)
        print()

        epoch_table.append((len(devices), args.batch_size, epoch_metrics.training_time, 1.0))
        args.batch_size *= 4

    return pd.DataFrame(epoch_table, columns=headers)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datapath', type=str, default='./data')
    parser.add_argument('-l', '--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('-m', '--momentum', type=float, default=0.9)
    parser.add_argument('-e', '--epochs', type=int, default=5, help="number of epochs to train")
    parser.add_argument('-w', '--workers', type=int, default=2, help="number of io workers")
    parser.add_argument('-c', '--cuda', action='store_true')
    parser.add_argument('--experiments', action='store_true',
                        help="run assignment experiments - ignores other program arguments")
    parser.add_argument('-s', '--summary', action='store_true', help='print resnet18 model summary for cifar10')
    parser.add_argument('-q', '--questions', action='store_true', help='answer model questions')

    args = parser.parse_args()

    if args.experiments:
        # one_gpu_results = increasing_batch_size(devices=[0])
        one_gpu_results = increasing_batch_size(devices=[0], batch_sizes=[32, 128])
        print(one_gpu_results)
        print('--------------------------\n')
        print('--------------------------\n')

        batch_sizes = list(one_gpu_results['batch_size'].unique())
        print(f'Using batch sizes {batch_sizes} for multi gpus...')
        print('--------------------------\n')
        print('--------------------------\n')

        two_gpu_results = increasing_batch_size(devices=[0, 1], batch_sizes=batch_sizes)
        two_gpu_results['speedup'] =  one_gpu_results['time'] / two_gpu_results['time']
        print(two_gpu_results)
        print('--------------------------\n')
        print('--------------------------\n')

        four_gpu_results = increasing_batch_size(devices=[0, 1, 2, 3], batch_sizes=batch_sizes)
        four_gpu_results['speedup'] = one_gpu_results['time'] / four_gpu_results['time']
        print(four_gpu_results)
        print('--------------------------\n')
        print('--------------------------\n')


    else:
        print(f'Running Ad Hoc Training...')

        training_args = TrainingArgs(
            devices=[0],
            datapath=args.datapath,
            workers=args.workers,
            learning_rate=args.lr,
            momentum=args.momentum,
            optimizer=args.optimizer,
            epochs=args.epochs
        )
        
        training_metrics = simple_training(training_args)
        training_metrics.summarize()

if __name__ == '__main__':
    main()
