
class TrainingArgs:
    def __init__(self, devices, datapath, workers, learning_rate, momentum, epochs, batch_size, warmup_epochs, weight_decay):
        self.devices = devices
        self.datapath = datapath
        self.workers = workers
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.epochs = epochs
        self.batch_size = batch_size
        self.warmup_epochs = warmup_epochs
        self.weight_decay = weight_decay
    
    def summarize(self):
        print('Training Args')
        print('Device: ', self.devices)
        print('Datapath: ', self.datapath)
        print('Workers: ', self.workers)
        print('LearningRate: ', self.learning_rate)
        print('Momentum: ', self.momentum)
        print('WeightDecay: ', self.weight_decay)
        print('Epochs: ', self.epochs)
        print('BatchSize: ', self.batch_size)
        print('WarmUpEpochs: ', self.warmup_epochs)


class EpochMetrics: 
    def __init__(self, dataloading_time, training_time, total_time, loss, acc):
        self.epoch = None
        self.dataloading_time = dataloading_time
        self.training_time = training_time
        self.total_time = total_time
        self.loss = loss
        self.acc = acc
    
    def __str__(self):
        epoch_info = f'Epoch {self.epoch} ' if self.epoch is not None else ''
        return f'{epoch_info}DataLoader Time: {self.dataloading_time}, Training Time: {self.training_time}, Total Time: {self.total_time}, Loss: {self.loss}, Acc: {self.acc}'


class TrainingMetrics:
    def __init__(self, training_args:TrainingArgs):
        self.args = training_args
        self.dataset_init_time = 0.0
        self.epoch_metrics = []
    
    def add_epoch_metrics(self, epoch_metrics:EpochMetrics):
        self.epoch_metrics.append(epoch_metrics)
    
    def total_io_time(self):
        return self.dataset_init_time
    
    def total_dataloader_time(self):
        return sum([epoch.dataloading_time for epoch in self.epoch_metrics])
    
    def total_training_time(self):
        return sum([epoch.training_time for epoch in self.epoch_metrics])
    
    def total_running_time(self):
        return sum([epoch.total_time for epoch in self.epoch_metrics])
    
    def avg_dataloader_time(self):
        return self.total_dataloader_time() / len(self.epoch_metrics)
    
    def avg_training_time(self):
        return self.total_training_time() / len(self.epoch_metrics)
    
    def avg_total_time(self):
        return self.total_running_time() / len(self.epoch_metrics)

    def summarize(self):
        self.args.summarize()
        print('Dataset Initialization Time: ', self.dataset_init_time)
        print(' Epoch Metrics: ')
        for epoch in self.epoch_metrics:
            print('   ', epoch)
