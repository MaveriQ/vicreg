# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from pathlib import Path
import argparse
import json
import math
import os
import sys
import time
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import nn, optim
import torch.distributed as dist
# import torchvision.datasets as datasets
import datasets

# import augmentations as aug
from distributed import init_distributed_mode
from torch.utils.tensorboard import SummaryWriter
# import resnet
import transformers
transformers.logging.set_verbosity_error()

def get_arguments():
    parser = argparse.ArgumentParser(description="Pretrain a resnet model with VICReg", add_help=False)

    # Data
    # parser.add_argument("--data-dir", type=Path, default="/path/to/imagenet", required=True,
    #                     help='Path to the image net dataset')
    parser.add_argument("--corpus", type=str, default="simcse", choices=['wikipedia','bookcorpus','simcse', 'test'],
                        help='Corpus to use for training. Default : wikipedia')
    parser.add_argument("--seq_len", type=int, default=128,
                        help='Sequence length for Transformer model')
        
    # Checkpoints
    parser.add_argument("--exp-dir", type=Path, default="/mounts/data/proj/jabbar/vicreg",
                        help='Path to the experiment folder, where all logs/checkpoints will be stored')
    parser.add_argument("--log-freq-time", type=int, default=60,
                        help='Print logs to the stats.txt file every [log-freq-time] seconds')
    parser.add_argument("--resume-from-checkpoint", action='store_true', 
                        help='Resumes from last checkpoint')

    # Model
    parser.add_argument("--arch", type=str, default="bert-base-uncased",
                        help='Architecture of the backbone encoder network')
    parser.add_argument("--mlp", default="8192-8192-8192",
                        help='Size and number of layers of the MLP expander head')

    # Optim
    parser.add_argument("--epochs", type=int, default=10,
                        help='Number of epochs')
    parser.add_argument("--warmup-epochs", type=int, default=2,
                        help='Number of warmup epochs for LR scheduler')
    parser.add_argument("--batch-size", type=int, default=64,
                        help='Effective batch size (per worker batch size is [batch-size] / world-size)')
    parser.add_argument("--base-lr", type=float, default=0.2,
                        help='Base learning rate, effective learning after warmup is [base-lr] * [batch-size] / 256')
    parser.add_argument("--wd", type=float, default=1e-6,
                        help='Weight decay')
    parser.add_argument("--grad-norm", type=float, default=1.0,
                        help='Gradient Norm for clipping')
    parser.add_argument("--grad-accum-steps", type=int, default=1,
                        help='Number of gradient accumulation steps')
    
    # Loss
    parser.add_argument("--sim-coeff", type=float, default=25.0,
                        help='Invariance regularization loss coefficient')
    parser.add_argument("--std-coeff", type=float, default=25.0,
                        help='Variance regularization loss coefficient')
    parser.add_argument("--cov-coeff", type=float, default=1.0,
                        help='Covariance regularization loss coefficient')

    # Running
    parser.add_argument("--exp-name", type=str, required=True,
                        help='Name of Exp to be passed to log dir')
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    # Distributed
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')

    return parser

def to_cuda(batch,gpu):
    
    for k,v in batch.items():
        batch[k] = v.cuda(gpu, non_blocking=True)
    
    return batch

def main(args):
    writer = SummaryWriter(args.exp_dir/'tb_logs'/args.exp_name)
    args.exp_dir = args.exp_dir/args.exp_name
    torch.backends.cudnn.benchmark = True
    init_distributed_mode(args)
    print(args)
    gpu = torch.device(args.device)

    if args.rank == 0:
        args.exp_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.exp_dir / "stats.txt", "a", buffering=1)
        print(" ".join(sys.argv))
        print(" ".join(sys.argv), file=stats_file)

    # transforms = aug.TrainTransform()

    # dataset = datasets.ImageFolder(args.data_dir / "train", transforms)
    if args.corpus=='bookcorpus':
        corpus = datasets.load_dataset(args.corpus,split='train')
    elif args.corpus=='wikipedia':
        corpus = datasets.load_dataset(args.corpus,name='20200501.en',split='train')
    elif args.corpus=='simcse':
        corpus = datasets.load_dataset('text', data_files='wiki1m_for_simcse.txt',split='train')
    elif args.corpus=='test':
        corpus = datasets.load_dataset('text', data_files='wiki1m_for_simcse.txt',split='train')
        corpus = corpus.select(range(1000))
        args.num_workers = 2
                
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = corpus.map(lambda e: tokenizer(e['text'],truncation=True,padding='max_length',max_length=args.seq_len),
                         remove_columns='text',num_proc=args.num_workers)
    dataset.set_format('torch')
    
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=per_device_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=sampler,
    )
    dataset_len = len(loader)

    model = VICReg(args).cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    optimizer = LARS(
        model.parameters(),
        lr=0,
        weight_decay=args.wd,
        weight_decay_filter=exclude_bias_and_norm,
        lars_adaptation_filter=exclude_bias_and_norm,
    )

    if (args.exp_dir / "model.pth").is_file() and args.resume_from_checkpoint:
        if args.rank == 0:
            print(f"resuming from checkpoint : {str(args.exp_dir/'model.pth')}")
        ckpt = torch.load(args.exp_dir / "model.pth", map_location="cpu")
        start_epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        start_epoch = 0

    start_time = last_logging = time.time()
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        for step, batch in enumerate(tqdm(loader,total=dataset_len), start=epoch * dataset_len):
            # x = x.cuda(gpu, non_blocking=True)
            # y = y.cuda(gpu, non_blocking=True)
            batch = to_cuda(batch,gpu)

            lr = adjust_learning_rate(args, optimizer, dataset_len, step)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss_dict = model(batch)
            
            for k,v in loss_dict.items():
                loss_dict[k] = scaler.scale(v)
            
            loss = loss_dict['loss']        
            scaler.scale(loss).backward()
            
            if (step + 1) % args.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
                
                scaler.step(optimizer)
                scaler.update()
            
            n_iter = step + epoch * dataset_len
            # if (n_iter + 1) % args.log_freq_time == 0 :
            #     writer.add_scalar('loss', loss.item(), n_iter)
            #     writer.add_scalar('repr_loss', loss_dict['repr_loss'].item(), n_iter)
            #     writer.add_scalar('cov_loss', loss_dict['cov_loss'].item(), n_iter)
            #     writer.add_scalar('std_loss', loss_dict['std_loss'].item(), n_iter)
            #     writer.add_scalar('lr', lr, n_iter)

            current_time = time.time()
            if args.rank == 0 and current_time - last_logging > args.log_freq_time:
                stats = dict(
                    epoch=epoch,
                    step=step,
                    loss=loss.item(),
                    time=int(current_time - start_time),
                    lr=lr,
                )
                writer.add_scalar('Loss/total', loss.item(), n_iter)
                writer.add_scalar('Loss/repr', loss_dict['repr_loss'].item(), n_iter)
                writer.add_scalar('Loss/cov', loss_dict['cov_loss'].item(), n_iter)
                writer.add_scalar('Loss/std', loss_dict['std_loss'].item(), n_iter)
                writer.add_scalar('lr', lr, n_iter)                
                # print(json.dumps(stats))
                print(json.dumps(stats), file=stats_file)
                last_logging = current_time
        if args.rank == 0:
            state = dict(
                epoch=epoch + 1,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
            )
            torch.save(state, args.exp_dir / "model.pth")
    if args.rank == 0:
        torch.save(model.module.backbone.state_dict(), args.exp_dir / f"{args.arch}.pth")


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


class VICReg(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_features = int(args.mlp.split("-")[-1])
        # self.backbone, self.embedding = resnet.__dict__[args.arch](
        #     zero_init_residual=True
        # )
        self.backbone = transformers.AutoModel.from_pretrained(args.arch)
        self.backbone.train()
        self.embedding = self.backbone.config.hidden_size
        self.projector = Projector(args, self.embedding)

    def forward(self, batch):
        x = self.projector(self.backbone(**batch).pooler_output)
        y = self.projector(self.backbone(**batch).pooler_output)

        repr_loss = F.mse_loss(x, y)

        x = torch.cat(FullGatherLayer.apply(x), dim=0)
        y = torch.cat(FullGatherLayer.apply(y), dim=0)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.args.batch_size - 1)
        cov_y = (y.T @ y) / (self.args.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        loss = (
            self.args.sim_coeff * repr_loss
            + self.args.std_coeff * std_loss
            + self.args.cov_coeff * cov_loss
        )
        return {'loss':loss,
                'repr_loss':repr_loss,
                'std_loss':std_loss,
                'cov_loss':cov_loss}


def Projector(args, embedding):
    mlp_spec = f"{embedding}-{args.mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)


def exclude_bias_and_norm(p):
    return p.ndim == 1


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class LARS(optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        weight_decay=0,
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=None,
        lars_adaptation_filter=None,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if g["weight_decay_filter"] is None or not g["weight_decay_filter"](p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if g["lars_adaptation_filter"] is None or not g[
                    "lars_adaptation_filter"
                ](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser('VICReg training script', parents=[get_arguments()])
    args = parser.parse_args()
    main(args)
