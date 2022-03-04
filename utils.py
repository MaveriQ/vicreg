import torch
import os
import math
import torch.distributed as dist

def exclude_bias_and_norm(p):
    return p.ndim == 1

def to_cuda(batch,gpu):
    
    for k,v in batch.items():
        batch[k] = v.cuda(gpu, non_blocking=True)
    
    return batch

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def batch_all_gather(x):
    x_list = FullGatherLayer.apply(x)
    return torch.cat(x_list, dim=0)

class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.environ["SLURM_JOB_ID"]}')
    exit()


def handle_sigterm(signum, frame):
    pass

def adjust_learning_rate(args, optimizer, dataset_len, step):
    max_steps = args.epochs * dataset_len
    warmup_steps = args.warmup_epochs * dataset_len
    base_lr = args.base_lr * args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

def alternate_learning_rate(named_params, lr_normal, lr_alternate, alternate_param="coeff"):
    params = []
    excluded_params = []

    for name, param in named_params:
        if not param.requires_grad:
            continue
        elif alternate_param in name:
            excluded_params.append(param)
        else:
            params.append(param)

    return [
        {"params": params, "lr": lr_normal},
        {"params": excluded_params, "lr": lr_alternate},
    ]