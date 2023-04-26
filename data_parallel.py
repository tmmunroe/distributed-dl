import torch
import time
from itertools import chain
from torch.nn import DataParallel
from torch.nn.parallel.scatter_gather import scatter_kwargs, gather
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.parallel_apply import parallel_apply

class DataParallelMetrics:
    def __init__(self):
        self.training_time = 0.0
        self.replicate_time = 0.0
        self.scatter_time = 0.0
        self.gather_time = 0.0
    
    @property
    def communication_time(self):
        return self.replicate_time + self.scatter_time + self.gather_time
    
    def total_time(self):
        return self.training_time + self.communication_time


class DataParallelWithMetrics(DataParallel):
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super().__init__(module, device_ids=device_ids, output_device=output_device, dim=dim)
        self.metrics = []
    
    def start_new_metrics(self):        
        self.metrics.append(DataParallelMetrics())

    def forward(self, *inputs, **kwargs):
        with torch.autograd.profiler.record_function("DataParallel.forward"):
            if not self.device_ids:
                start = time.perf_counter()
                outputs = self.module(*inputs, **kwargs)
                self.metrics[-1].training_time += time.perf_counter() - start
                return outputs
                
            for t in chain(self.module.parameters(), self.module.buffers()):
                if t.device != self.src_device_obj:
                    raise RuntimeError("module must have its parameters and buffers "
                                       "on device {} (device_ids[0]) but found one of "
                                       "them on device: {}".format(self.src_device_obj, t.device))

            inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
            # for forward function without any inputs, empty list and dict will be created
            # so the module can be executed on one device which is the first one in device_ids
            if not inputs and not kwargs:
                inputs = ((),)
                kwargs = ({},)

            if len(self.device_ids) == 1:
                start = time.perf_counter()
                outputs = self.module(*inputs[0], **kwargs[0])
                self.metrics[-1].training_time += time.perf_counter() - start
                return outputs

            replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
            outputs = self.parallel_apply(replicas, inputs, kwargs)
            return self.gather(outputs, self.output_device)

    def replicate(self, module, device_ids):
        start = time.perf_counter()
        out = replicate(module, device_ids, not torch.is_grad_enabled())
        self.metrics[-1].replicate_time += time.perf_counter() - start
        return out

    def scatter(self, inputs, kwargs, device_ids):
        start = time.perf_counter()
        out = scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)
        self.metrics[-1].scatter_time += time.perf_counter() - start
        return out

    def parallel_apply(self, replicas, inputs, kwargs):
        start = time.perf_counter()
        out = parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])
        self.metrics[-1].training_time += time.perf_counter() - start
        return out

    def gather(self, outputs, output_device):
        start = time.perf_counter()
        out = gather(outputs, output_device, dim=self.dim)
        self.metrics[-1].gather_time += time.perf_counter() - start
        return out
