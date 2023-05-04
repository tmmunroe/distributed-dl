import argparse
import time
from collections import namedtuple
from tabulate import tabulate
import pandas as pd
from typing import Tuple

import math

import torch
import torch.cuda
import torch.nn as nn
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data.dataloader import DataLoader
from data_parallel import DataParallelWithMetrics, DataParallelMetrics
from torchvision.models import resnet18

from utils import TrainingArgs, TrainingMetrics, EpochMetrics

def default_training_args():
    return TrainingArgs(
        devices=[0],
        datapath='./data',
        workers=2,
        learning_rate=0.1,
        momentum=0.9,
        weight_decay=5e-4,
        epochs=1,
        batch_size=32,
        warmup_epochs=None
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
        # print(type(inputs[0]), inputs[0], type(inputs[0]), inputs[0].dtype)
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

def dataset_size_in_bytes(args: TrainingArgs):
    training_data = datasets.CIFAR10(root=args.datapath, train=True, download=True)
    image_dim = 3*32*32
    float_size_bytes = 4
    return len(training_data)*image_dim*float_size_bytes

def simple_training(args: TrainingArgs) -> Tuple[TrainingMetrics, DataParallelWithMetrics]:
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

    train_dataloader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Create ResNet=18 Model and training parameters
    nn_model = resnet18()
    if len(args.devices) > 1:
        nn_model = DataParallelWithMetrics(nn_model, device_ids=args.devices)
    nn_model.to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(nn_model.parameters(), lr=args.learning_rate,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = None
    if args.warmup_epochs:
        lr_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1/3,
                                                   total_iters=args.warmup_epochs,
                                                   verbose=True)
    
    training_metrics = TrainingMetrics(args)
    training_metrics.dataset_init_time = dataset_init_elapsed

    if device.type == 'cuda': # make sure gpu ops are done before timing
        torch.cuda.synchronize()
    for epoch in range(args.epochs):
        print(f"Epoch: {epoch+1} of {args.epochs}")

        if hasattr(nn_model, 'start_new_metrics'):
            nn_model.start_new_metrics() # to track more precise times internally for communication and training

        epoch_metrics = train(nn_model, optimizer, loss_func, train_dataloader, device)
        epoch_metrics.epoch = epoch+1
        training_metrics.add_epoch_metrics(epoch_metrics)
        if lr_scheduler:
            lr_scheduler.step()

    return training_metrics, nn_model

def batch_size_generator():
    batch_size = 32
    while True:
        yield batch_size
        batch_size *= 4

def increasing_batch_size(devices:list, batch_sizes_per_gpu:list=None):
    #Q1, Q2, Q3.1
    print(devices)

    args = default_training_args()
    args.epochs = 2
    args.devices = devices

    if batch_sizes_per_gpu is None:
        print('Using unbounded batch size per gpu generator')
        batch_sizes_per_gpu = batch_size_generator()

    headers = ['gpus', 'batch_size_per_gpu', 'total_batch_size', 'time', 'parallel_apply', 'replicate', 'scatter', 'gather']
    epoch_table = []
    for batch_size_per_gpu in batch_sizes_per_gpu:
        # release cuda memory for next batch_size experiment
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        try:
            batch_size = batch_size_per_gpu*len(devices)
            print('BatchSize: ', batch_size, '... BatchSizePerGpu: ', batch_size_per_gpu)

            args.batch_size = int(batch_size)
            training_metrics, nn_model = simple_training(args)
            if len(training_metrics.epoch_metrics) != 2:
                raise ValueError(f'Expected 2 epochs to be run.. but {len(epoch_metrics)} were run')
            if len(devices) > 1 and len(nn_model.metrics) != len(training_metrics.epoch_metrics):
                raise ValueError(f'Expected metrics for 2 epochs to be in nn_model.metrics.. but {len(nn_model.metrics)} were there')

            epoch_metrics:EpochMetrics = training_metrics.epoch_metrics[-1]

            if len(devices) > 1:
                data_parallel_metrics:DataParallelMetrics = nn_model.metrics[-1]
            else:
                data_parallel_metrics = DataParallelMetrics() # all zeroes
            
            print('Epoch 2 Metrics:')
            print(epoch_metrics)
            print()

            epoch_table.append((
                len(devices),
                batch_size_per_gpu,
                args.batch_size,
                epoch_metrics.training_time,
                data_parallel_metrics.training_time,
                data_parallel_metrics.replicate_time,
                data_parallel_metrics.scatter_time,
                data_parallel_metrics.gather_time
            ))
        except Exception as ex:
            print(f'Failed at batch_size_per_gpu {batch_size_per_gpu} with error: ')
            print(str(ex))
            break

    return pd.DataFrame(epoch_table, columns=headers)


def train_batch_size_simple(batch_size_per_gpu:int, devices:list, epochs:int):
    #Q4.1
    args = default_training_args()
    args.epochs = epochs
    args.devices = devices
    args.batch_size = int(batch_size_per_gpu*len(devices))

    headers = ['gpus', 'batch_size_per_gpu', 'total_batch_size', 'epoch', 
               'time', 'accuracy', 'loss']
    training_metrics, _ = simple_training(args)
    epoch:EpochMetrics = training_metrics.epoch_metrics[-1]
    epoch_table = [
        (len(args.devices), batch_size_per_gpu, args.batch_size, len(training_metrics.epoch_metrics), epoch.training_time,epoch.acc, epoch.loss)
    ]
    return pd.DataFrame(epoch_table, columns=headers)



def train_batch_size_with_remedies(batch_size_per_gpu:int, devices:list, epochs:int):
    #Q4.2
    args = default_training_args()
    base_batch_size = 128 # because we are comparing to hw2

    args.epochs = epochs
    args.devices = devices
    args.batch_size = int(batch_size_per_gpu*len(devices))

    # remedy 1 - linear scaling of learning rate with batch size
    learning_rate_scale = args.batch_size / base_batch_size
    args.learning_rate *= learning_rate_scale

    # remedy 2 - warm up
    args.warmup_epochs = 3


    print('Scaling learning rate by: ', learning_rate_scale)
    print('Using scaled learning rate: ', args.learning_rate)
    print('Using warmup epochs: ', args.warmup_epochs)
    
    headers = ['gpus', 'batch_size_per_gpu', 'total_batch_size', 'epoch',
               'warmup_epochs', 'learning_rate', 'time', 'accuracy', 'loss']
    training_metrics, _ = simple_training(args)
    epoch:EpochMetrics = training_metrics.epoch_metrics[-1]
    epoch_table = [
        (len(args.devices), batch_size_per_gpu, args.batch_size, len(training_metrics.epoch_metrics),
         args.warmup_epochs, args.learning_rate, epoch.training_time,epoch.acc, epoch.loss)
    ]
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
        nbytes_training = dataset_size_in_bytes(args)
        print('Training Data Bytes: ', nbytes_training)
        batch_sizes_per_gpu = None
        one_gpu_results = increasing_batch_size(devices=[0], batch_sizes_per_gpu=batch_sizes_per_gpu)
        # one_gpu_results['communication_time'] = one_gpu_results[['replicate', 'scatter', 'gather']].sum(axis=1)
        one_gpu_results['communication_time'] = 0
        one_gpu_results['computation_time'] = one_gpu_results['time'] - one_gpu_results['communication_time']
        one_gpu_results['speedup'] = 1.0
        keep_cols = 'gpus,batch_size_per_gpu,time,communication_time,computation_time,speedup'.split(',')
        print(tabulate(one_gpu_results[keep_cols], headers=keep_cols))
        # print(one_gpu_results)
        print('--------------------------\n')
        print('--------------------------\n')

        batch_sizes_per_gpu = list(one_gpu_results['batch_size_per_gpu'].unique())
        print(f'Using batch sizes per gpu {batch_sizes_per_gpu} for multi gpus...')
        print('--------------------------\n')
        print('--------------------------\n')

        two_gpu_results = increasing_batch_size(devices=[0, 1], batch_sizes_per_gpu=batch_sizes_per_gpu)
        two_gpu_results['computation_time'] = one_gpu_results['time'] / 2 # assume perfect scaling
        two_gpu_results['communication_time'] = two_gpu_results['time'] - two_gpu_results['computation_time']
        two_gpu_results['speedup'] = one_gpu_results['time'] / two_gpu_results['time']
        keep_cols = 'gpus,batch_size_per_gpu,time,communication_time,computation_time,speedup'.split(',')
        print(tabulate(two_gpu_results[keep_cols], headers=keep_cols))
        # print(two_gpu_results)
        print('--------------------------\n')
        print('--------------------------\n')

        four_gpu_results = increasing_batch_size(devices=[0, 1, 2, 3], batch_sizes_per_gpu=batch_sizes_per_gpu)
        four_gpu_results['computation_time'] = one_gpu_results['time'] / 4 # assume perfect scaling
        four_gpu_results['communication_time'] = four_gpu_results['time'] - four_gpu_results['computation_time']
        four_gpu_results['speedup'] = one_gpu_results['time'] / four_gpu_results['time']
        keep_cols = 'gpus,batch_size_per_gpu,time,communication_time,computation_time,speedup'.split(',')
        print(tabulate(four_gpu_results[keep_cols], headers=keep_cols))
        # print(four_gpu_results)
        print('--------------------------\n')
        print('--------------------------\n')

        gpus = 2
        bandwidth_bytes = nbytes_training * 2 * (gpus-1) / gpus
        two_gpu_results['total_gb_comm'] =  bandwidth_bytes * 1e-9
        two_gpu_results['bandwidth_utilization_amort'] =  two_gpu_results['total_gb_comm'] / two_gpu_results['time']
        two_gpu_results['bandwidth_utilization_comm'] =  two_gpu_results['total_gb_comm'] / two_gpu_results['communication_time']
        keep_cols = 'gpus,batch_size_per_gpu,time,communication_time,total_gb_comm,bandwidth_utilization_amort,bandwidth_utilization_comm'.split(',')
        print(tabulate(two_gpu_results[keep_cols], headers=keep_cols))

        print('--------------------------\n')
        print('--------------------------\n')

        gpus = 4
        bandwidth_bytes = nbytes_training * 2 * (gpus-1) / gpus
        four_gpu_results['total_gb_comm'] =  bandwidth_bytes * 1e-9
        four_gpu_results['bandwidth_utilization_amort'] =  four_gpu_results['total_gb_comm'] / four_gpu_results['time']
        four_gpu_results['bandwidth_utilization_comm'] =  four_gpu_results['total_gb_comm'] / four_gpu_results['communication_time']
        keep_cols = 'gpus,batch_size_per_gpu,time,communication_time,total_gb_comm,bandwidth_utilization_amort,bandwidth_utilization_comm'.split(',')
        print(tabulate(four_gpu_results[keep_cols], headers=keep_cols))


        print('--------------------------\n')
        print('--------------------------\n')


        max_batch_size_per_gpu = max(batch_sizes_per_gpu)

        large_batch_training = train_batch_size_simple(batch_size_per_gpu=max_batch_size_per_gpu, devices=[0, 1, 2, 3], epochs=5)
        print('Large Batch Training Results:')
        keep_cols = ['gpus', 'batch_size_per_gpu', 'total_batch_size', 'epoch', 
                    'time', 'accuracy', 'loss']
        print(tabulate(large_batch_training[keep_cols], headers=keep_cols))
        print('--------------------------\n')
        print('--------------------------\n')

        remedies_training = train_batch_size_with_remedies(batch_size_per_gpu=max_batch_size_per_gpu, devices=[0, 1, 2, 3], epochs=5)
        print('Large Batch Training with Warmup and Linearly Scaled Learning Rate Results:')
        keep_cols = ['gpus', 'batch_size_per_gpu', 'total_batch_size', 'epoch',
                    'warmup_epochs', 'learning_rate', 'time', 'accuracy', 'loss']
        print(tabulate(remedies_training[keep_cols], headers=keep_cols))
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
            epochs=args.epochs,
            batch_size=32,
            warmup_epochs=None
        )
        
        training_metrics, nn_model = simple_training(training_args)
        training_metrics.summarize()

if __name__ == '__main__':
    main()
