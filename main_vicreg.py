# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from pathlib import Path
import argparse
import json
import sys
import time
from tqdm import tqdm
import torch
from torch import nn
# import torchvision.datasets as datasets
import datasets

# import augmentations as aug
from distributed import init_distributed_mode
from torch.utils.tensorboard import SummaryWriter
# import resnet
import transformers
transformers.logging.set_verbosity_error()

from utils import exclude_bias_and_norm, adjust_learning_rate, to_cuda, alternate_learning_rate
from models import VICReg, LARS

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
    parser.add_argument("--mlp", default="1024",
                        help='Size and number of layers of the MLP expander head')

    # Optim
    parser.add_argument("--epochs", type=int, default=10,
                        help='Number of epochs')
    parser.add_argument("--warmup-epochs", type=float, default=1.0,
                        help='Number of warmup epochs for LR scheduler')
    parser.add_argument("--batch-size", type=int, default=64,
                        help='Effective batch size (per worker batch size is [batch-size] / world-size)')
    parser.add_argument("--base-lr", type=float, default=1e-3,
                        help='Base learning rate, effective learning after warmup is [base-lr] * [batch-size] / 256')
    parser.add_argument("--wd", type=float, default=1e-6,
                        help='Weight decay')
    parser.add_argument("--grad-norm", type=float, default=1.0,
                        help='Gradient Norm for clipping')
    parser.add_argument("--grad-accum-steps", type=int, default=1,
                        help='Number of gradient accumulation steps')
    parser.add_argument("--use-alternate-lr", action='store_true', 
                        help='Alternate LR for weight ratios')
    
    # Loss
    parser.add_argument("--sim-coeff", type=float, default=25.0,
                        help='Invariance regularization loss coefficient')
    parser.add_argument("--std-coeff", type=float, default=25.0,
                        help='Variance regularization loss coefficient')
    parser.add_argument("--cov-coeff", type=float, default=1.0,
                        help='Covariance regularization loss coefficient')
    parser.add_argument("--use-param-weights", action='store_true', 
                        help='Use learnable weights for the three losses')
    
    # Running
    parser.add_argument("--exp-name", type=str, required=True,
                        help='Name of Exp to be passed to log dir')
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    # Distributed
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')

    return parser

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
    
    if args.use_alternate_lr:
        params = alternate_learning_rate(model.named_parameters(),args.base_lr,args.base_lr*10)
    else: 
        params = model.parameters()
        
    optimizer = LARS(
        params, # model.parameters(),
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
        
        if args.rank == 0:
            iter = tqdm(loader,total=dataset_len)
        else: 
            iter = loader
            
        for step, batch in enumerate(iter, start=epoch * dataset_len):
            
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

            current_time = time.time()
            
            if args.rank == 0:
                iter.set_description(f"Epoch {epoch}")
                iter.set_postfix({'lr':lr,
                                  'loss':loss.item()})
                if current_time - last_logging > args.log_freq_time:
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
                    writer.add_scalars('Parameters',model.module.param, n_iter)                
                    writer.add_scalar('lr', lr, n_iter)                
                    
                    print(json.dumps(stats), file=stats_file)
                    last_logging = current_time
        if args.rank == 0:
            state = dict(
                epoch=epoch + 1,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
            )
            torch.save(state, args.exp_dir / f"ckpt_epoch_{epoch+1}.pth")
    if args.rank == 0:
        torch.save(model.module.backbone.state_dict(), args.exp_dir / f"{args.arch}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser('VICReg training script', parents=[get_arguments()])
    args = parser.parse_args()
    main(args)
