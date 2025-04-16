# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
import time
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    teacher: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    log_writer=None,
                    args=None):
    """
    单epoch训练流程

    参数:
        model: 学生模型，需要进行梯度更新的模型
        teacher: 教师模型，仅用于推理（不更新参数）
        data_loader: 数据加载器，提供(batch_size, [view1, view2], labels)格式数据
        device: 训练设备（cuda/cpu）
        epoch: 当前epoch序号
        loss_scaler: 混合精度训练的梯度缩放器
        log_writer: TensorBoard日志记录器
        args: 训练配置参数
    """
    # 记录 epoch 开始时间
    epoch_start_time = time.time()

    # 训练模式设置
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'  # 进度条标题
    print_freq = 20  # 日志打印频率

    accum_iter = args.accum_iter  # 梯度累积步数

    optimizer.zero_grad()  # 初始化梯度

    # 日志目录打印（主进程）
    if log_writer is not None:
        print(f'log_dir: {log_writer.log_dir}')

    # 当前各loss权重数组，依次为: qk, vv, dyt, dyt_f, cls, out
    loss_weight = args.loss_weight

    # 开始迭代数据
    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # ==================== 学习率调整 ====================
        # 基于迭代次数的学习率调整（非epoch级别）
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer,  # 优化器对象
                data_iter_step / len(data_loader) + epoch,  # 归一化的训练进度
                args  # 参数配置
            )

        # ==================== 数据准备 ====================
        # samples结构应为 [teacher_view_tensor, student_view_tensor]
        # 获取输入形状参数
        B, C, H, W = samples[0].shape  # Batch, Channels, Height, Width
        N = B  # 样本数量
        L = (H // 16) * (W // 16)  # Patch数量（假设16x16分块）

        # 生成随机噪声
        noise = torch.rand(N, L, device=samples[0].device)  # [0,1]均匀分布噪声

        # ==================== 前向计算 ====================
        with torch.cuda.amp.autocast():  # 混合精度上下文
            # 教师模型推理（不计算梯度）
            with torch.no_grad():
                teacher_out = teacher(samples[0].to(device, non_blocking=True))  # 教师视图前向

            # 学生模型前向，用的large18/24层和base12/12层做损失
            qk_loss, vv_loss, dyt_loss, dyt_f_loss, cls_loss = model(
                samples[1].to(device, non_blocking=True),  # 学生视图输入
                teacher_out  # 教师模型输出作为监督信号
            )

        # 总损失计算（加权）
        loss = (loss_weight[0] * qk_loss +
                loss_weight[1] * vv_loss +
                loss_weight[2] * dyt_loss +
                loss_weight[3] * dyt_f_loss +
                loss_weight[4] * cls_loss
                )
        loss_value = loss.item()  # 获取标量损失值

        # ==================== 损失检查 ====================
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        # ==================== 反向传播 ====================
        loss /= accum_iter  # 梯度累积的损失归一化
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0
        )

        # 梯度清零策略
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()  # 同步CUDA操作

        # ==================== 指标记录 ====================
        metric_logger.update(total_loss=loss.item())

        # 根据各loss的权重条件记录
        if loss_weight[0] != 0:
            metric_logger.update(qk_loss=qk_loss.item())
        if loss_weight[1] != 0:
            metric_logger.update(vv_loss=vv_loss.item())
        if loss_weight[2] != 0:
            metric_logger.update(dyt_loss=dyt_loss.item())
        if loss_weight[3] != 0:
            metric_logger.update(dyt_f_loss=dyt_f_loss.item())
        if loss_weight[4] != 0:
            metric_logger.update(cls_loss=cls_loss.item())

        # 记录学习率指标
        lr_current = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr_current)

        # ==================== TensorBoard日志记录 ====================
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value, epoch_1000x)
            log_writer.add_scalar('lr', lr_current, epoch_1000x)
            if loss_weight[0] != 0:
                log_writer.add_scalar('qk_loss', qk_loss.item(), epoch_1000x)
            if loss_weight[1] != 0:
                log_writer.add_scalar('vv_loss', vv_loss.item(), epoch_1000x)
            if loss_weight[2] != 0:
                log_writer.add_scalar('dyt_loss', dyt_loss.item(), epoch_1000x)
            if loss_weight[3] != 0:
                log_writer.add_scalar('dyt_f_loss', dyt_f_loss.item(), epoch_1000x)
            if loss_weight[4] != 0:
                log_writer.add_scalar('cls_loss', cls_loss.item(), epoch_1000x)

    # ==================== epoch结束处理 ====================
    metric_logger.synchronize_between_processes()
    averaged_stats = metric_logger.meters
    print("Averaged stats:", metric_logger)

    # 计算并记录本 epoch 耗时
    epoch_time = time.time() - epoch_start_time
    print(f"Epoch {epoch} finished, time: {epoch_time:.2f} seconds")
    if log_writer is not None:
        log_writer.add_scalar("epoch_time", epoch_time, epoch)

    # 返回指标字典（可加入 epoch_time 用于后续分析）
    averaged_stats["epoch_time"] = epoch_time
    return {k: meter.global_avg if hasattr(meter, "global_avg") else meter for k, meter in averaged_stats.items()}
