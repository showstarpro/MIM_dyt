# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

# Standard library imports
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
from PIL import Image

# PyTorch core components
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Image model libraries
import timm

assert timm.__version__ == "0.3.2"  # Verify compatible timm version
import timm.optim.optim_factory as optim_factory

# Custom utilities
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

# Model architectures
import models_tinymim
from engine_pretrain import train_one_epoch
import models_teacher


def get_args_parser():
    """Defines and configures all command-line arguments for training"""
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)

    # Training hyperparameters
    parser.add_argument('--batch_size', default=4, type=int,
                        help='Per-GPU batch size (effective batch = batch_size * accum_iter * num_gpus)')
    parser.add_argument('--epochs', default=100, type=int,
                        help='Total number of training epochs')
    parser.add_argument('--start_epoch', default=0, type=int, # 加了一个start_epochs，本来没有但代码里有用
                        help='start of training epochs')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Number of gradient accumulation steps')

    # Model configuration
    parser.add_argument('--model', default='tinymim_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of student model architecture to train')
    parser.add_argument('--norm', default='norm', type=str,
                        help='norm or dyt')
    parser.add_argument('--input_size', default=224, type=int,
                        help='Input image resolution (square size)')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay coefficient for regularization')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='Absolute learning rate (overrides blr if set)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='Base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='Minimum learning rate for schedulers')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='Number of warmup epochs for learning rate')

    # Dataset configuration
    parser.add_argument('--data_path', default='lpai/dataset/imagenet-1k/0-1-0', type=str,
                        help='Root directory path for dataset')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='Output directory for saving checkpoints')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='Directory for TensorBoard logs')

    # System configuration
    parser.add_argument('--device', default='cuda',
                        help='Compute device to use (cuda/cpu)')
    parser.add_argument('--seed', default=0, type=int,
                        help='Random seed for reproducibility')
    parser.add_argument('--resume', default='',
                        help='Path to checkpoint for resuming training')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='Number of data loading workers')

    # Memory optimization
    parser.add_argument('--pin_mem', action='store_true',
                        help='Enable pinned memory for faster data transfer to GPU')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='Automatic argument for distributed launch')
    parser.add_argument('--dist_on_itp', action='store_true',
                        help='Enable distributed training on ITP clusters')
    parser.add_argument('--dist_url', default='env://',
                        help='URL used to initialize distributed training')

    # Teacher model configuration MIM_dyt-master/MIM_dyt-master/
    parser.add_argument("--teacher_path", default="mae_pretrain_vit_large.pth",type=str,
                        help='File path to pre-trained teacher model weights')
    parser.add_argument("--teacher_model", default = "mae_vit_large",type=str,
                        help='Architecture name of teacher model')

    # 做损失的层数
    parser.add_argument('--intermediate', default=[2,4,6,8,10],nargs='+', type=int,    # teacher层数
                        help='Layer index for teacher feature distillation')
    parser.add_argument('--layer', default=[1,3,5, 7, 9],nargs='+', type=int,           # student层数
                        help='Layer index for student feature distillation')

    # 损失的权重
    parser.add_argument('--loss_weight', default=[0,0,1,1,1],nargs='+', type=int,  # qk，vv，dyt，dyt_f，cls
                        help='the loss weight with qk vv dyt dyt_f cls')

    return parser


class TwoCropsTransform:
    """Data augmentation that generates two views from one image"""

    def __init__(self, common_transform, teacher_transform, student_transform2):
        """
        Args:
            common_transform: Shared transforms for both views
            teacher_transform: Transform specific to teacher model input
            student_transform2: Transform specific to student model input
        """
        self.common_transform = common_transform
        self.teacher_transform = teacher_transform
        self.student_transform2 = student_transform2

    def __call__(self, x):
        """Apply transforms to generate paired augmented views"""
        x = self.common_transform(x)
        return [self.teacher_transform(x), self.student_transform2(x)]


def main(args):
    """Main training procedure"""

    # Initialize distributed training backend
    misc.init_distributed_mode(args)

    # Print execution environment information
    print(f'Job directory: {os.path.dirname(os.path.realpath(__file__))}')
    print(f'Training arguments:\n{args}')

    # Set computation device
    device = torch.device(args.device)

    # Seed all RNGs for reproducibility
    seed = args.seed + misc.get_rank()  # Add rank to seed for distributed safety
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True  # Enable cuDNN auto-tuner

    # Data augmentation pipelines -------------------------------------------------
    # Common transforms applied to both views
    common_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),
        transforms.RandomHorizontalFlip()
    ])

    # Teacher-specific transforms (normalization only)
    teacher_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Student-specific transforms (additional augmentations)
    student_transform = transforms.Compose([
        transforms.ColorJitter(0.4, 0.4, 0.4),  # Color augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset preparation ---------------------------------------------------------
    # Synthetic dataset for testing (comment out for real data)
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'),
                                         transform=TwoCropsTransform(common_transform, teacher_transform,
                                                                     student_transform))
    print(dataset_train)
    # class FakeImageNet(Dataset):
    #     def __init__(self, size=224, num_samples=1000):
    #         self.size = size
    #         self.num_samples = num_samples
    #         self.classes = ['class_{}'.format(i) for i in range(1000)]
    #         self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
    #
    #         # 生成符合TwoCropsTransform格式的数据
    #         self.samples = [
    #             (
    #                 # 模拟TwoCropsTransform的输出：包含两个视图的列表
    #                 [
    #                     self._generate_image(),  # 教师视图
    #                     self._generate_image()  # 学生视图
    #                 ],
    #                 np.random.randint(0, 1000)  # 标签
    #             )
    #             for _ in range(num_samples)
    #         ]
    #
    #     def _generate_image(self):
    #         """生成随机PIL图像"""
    #         return Image.fromarray(np.random.randint(0, 255, (self.size, self.size, 3), dtype=np.uint8))
    #
    #     def __getitem__(self, index):
    #         """返回格式: ( [view1_tensor, view2_tensor], label ) """
    #         views, label = self.samples[index]
    #
    #         processed_views = [
    #             transforms.ToTensor()(view) for view in views
    #         ]
    #
    #         return processed_views, label  # 返回二元组(views, label)
    #
    #     def __len__(self):
    #         return self.num_samples
    #
    #
    # dataset_train = FakeImageNet()
    # print(f'Dataset information:\n{dataset_train}')

    # Distributed sampler configuration
    if True:  # Always use distributed mode in this setup
        num_tasks = misc.get_world_size()  # Total number of processes
        global_rank = misc.get_rank()  # Current process rank
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train,
            num_replicas=num_tasks,
            rank=global_rank,
            shuffle=True
        )
        print(f"Distributed sampler: {sampler_train}")
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    # Initialize TensorBoard logger (master process only)
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # Create data loader with configured parameters
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,  # Drop incomplete batches
    )

    # Model initialization --------------------------------------------------------
    # Student model instantiation
    model = models_tinymim.__dict__[args.model](norm = args.norm,
                                                last_head = 16 if args.teacher_model =="mae_vit_large" else 12,
                                                tea_embed_dim = 1024 if args.teacher_model =="mae_vit_large" else 768,
                                                )  # Dynamic model loading
    model.layer = args.layer

    # Teacher model setup
    teacher = models_teacher.__dict__[args.teacher_model]()  # Load teacher architecture
    teacher.load_state_dict(torch.load(args.teacher_path, map_location="cpu", weights_only=True)["model"])
    teacher.intermediate = args.intermediate
    teacher.eval()  # Freeze teacher parameters

    # Device placement
    model.to(device)
    teacher.to(device)
    model_without_ddp = model  # Reference for non-distributed model
    print(f"Student model architecture:\n{model_without_ddp}")

    # Learning rate calculation ---------------------------------------------------
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # Auto-scale learning rate
        args.lr = args.blr * eff_batch_size / 256

    # Print learning rate information
    print(f"Base learning rate: {args.lr * 256 / eff_batch_size:.2e}")
    print(f"Actual learning rate: {args.lr:.2e}")
    print(f"Gradient accumulation steps: {args.accum_iter}")
    print(f"Effective batch size: {eff_batch_size}")

    # Distributed Data Parallel (DDP) configuration
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=True  # For complex architectures
        )
        model_without_ddp = model.module  # Access base model

    # Optimizer configuration -----------------------------------------------------
    # Apply weight decay excluding bias and normalization layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.lr,
        betas=(0.9, 0.95)  # Optimizer momentum parameters
    )
    loss_scaler = NativeScaler()  # Mixed precision gradient scaling

    # Checkpoint auto-resume mechanism --------------------------------------------
    if os.path.exists(args.output_dir):
        ckpt_files = os.listdir(args.output_dir)
        if ckpt_files:
            # Find latest checkpoint in output directory
            for i in range(800):
                checkpoint_name = f"checkpoint-{i}.pth"
                if checkpoint_name in ckpt_files:
                    args.resume = str(Path(args.output_dir) / checkpoint_name)
                    print(f"Auto-resuming from checkpoint: {args.resume}")

    # Load checkpoint if available
    misc.load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler
    )

    # Training loop ---------------------------------------------------------------
    print(f"Starting training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)  # Shuffle distributed data

        # Train for one epoch
        train_stats = train_one_epoch(
            model,
            teacher,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            log_writer=log_writer,
            args=args
        )

        # Save checkpoint periodically
        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch
            )

        # Log training statistics
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            'epoch': epoch,
        }
        if args.output_dir and misc.is_main_process():
            with open(Path(args.output_dir) / "log.txt", "a") as f:
                f.write(json.dumps(log_stats) + "\n")

    # Training duration statistics
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Total training time: {total_time_str}')


if __name__ == '__main__':
    # Parse command-line arguments and launch main training
    args = get_args_parser().parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)