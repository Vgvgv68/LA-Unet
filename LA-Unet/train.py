"""
医学图像分割训练脚本（UNet架构）
版本：1.0 | 2025-04-08
"""
from unet import UNet
from attunet import AttU_Net
from malunet import MALUNet
from ukan import archs
from UNet2Plus.UNet2Plus import UNet2Plus
from unext.model import UNext
from egeunet.egeunet import EGEUNet
from v3.mobilenetv3 import  MobileNetV3
# from lvunet.LV_UNet import LV_UNet
#from UNet_v2.UNet_v2 import UNetV2
from dpt.dpt import DPT
# ==================== 标准库导入 ====================
import os
import argparse
import random
from glob import glob
from collections import OrderedDict

# ==================== 第三方库导入 ====================
import json
import math
import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from albumentations import (
    RandomRotate90, Resize, HorizontalFlip, VerticalFlip,ElasticTransform,
    RandomBrightnessContrast, CLAHE, GridDistortion, CoarseDropout,
    RandomGamma, RandomGridShuffle, ColorJitter
)
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split, KFold
from thop import profile
from torch.amp import GradScaler
from torch import autocast
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch_ema import ExponentialMovingAverage


# ==================== 本地模块导入 ====================
import losses
from dataset import Dataset
from metrics import iou_score
import utils
from utils import AverageMeter, str2bool, WarmupCosineSchedule
from LA_UNet import LA_UNet

# ==================== 全局配置 ====================
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'  # 禁用Albumentations自动更新（避免版本冲突）
os.environ['ALBUMENTATIONS_NO_UPDATE_CHECK'] = '1'

LOSS_NAMES = losses.__all__ + ['FocalBCEDiceLoss']  # 包含自定义和PyTorch内置损失

# ================================================

def parse_args():
    """解析命令行参数并返回配置对象

    Returns:
        argparse.Namespace: 包含所有配置参数的对象
    """
    parser = argparse.ArgumentParser(
        description='医学图像分割训练配置',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # 显示默认值
    )

    # -------------------- 基础配置 --------------------
    base_group = parser.add_argument_group('Base Configuration')
    base_group.add_argument('--cfg', type=str, metavar="FILE",
                            help='YAML配置文件路径 (覆盖命令行参数)')
    base_group.add_argument('--name', default='busi_DPT_woDS',
                            help='实验名称 (默认: arch+timestamp)')
    base_group.add_argument('--num_workers', default=7, type=int,
                            help='数据加载线程数')

    # -------------------- 模型配置 --------------------
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--arch', default='DPT', type=str,
                             help='模型架构名称')
    model_group.add_argument('--deep_supervision', default=False, type=str2bool,
                             help='是否使用深度监督')
    model_group.add_argument('--input_channels', default=3, type=int,
                             help='输入图像通道数')
    model_group.add_argument('--input_w', default=256, type=int,
                             help='输入图像宽度')
    model_group.add_argument('--input_h', default=256, type=int,
                             help='输入图像高度')
    model_group.add_argument('--num_classes', default=1, type=int,
                             help='分割类别数')

    # -------------------- 数据配置 --------------------
    data_group = parser.add_argument_group('Dataset Configuration')
    data_group.add_argument('--dataset', default='isic',
                            help='数据集名称,isic(png),busi(png),cvc_clinicdb(tif),CVC_ColonDB(tiff),Kvasir_SEG(jpg)')
    data_group.add_argument('--img_ext', default='.png',
                            help='图像文件扩展名')
    data_group.add_argument('--mask_ext', default='.png',
                            help='标注文件扩展名')

    # -------------------- 训练配置 --------------------
    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument('--epochs', default=120, type=int, metavar='N',
                             help='总训练轮次')
    train_group.add_argument('-b', '--batch_size', default=8, type=int, metavar='N',
                             help='批次大小 (默认: 8)')
    train_group.add_argument('--loss', default='FocalBCEDiceLoss', choices=LOSS_NAMES,
                             help='损失函数: ' + ' | '.join(LOSS_NAMES))

    # -------------------- 优化器配置 --------------------
    optim_group = parser.add_argument_group('Optimizer Configuration')
    optim_group.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'],
                             help='优化器选择')
    optim_group.add_argument('--lr', '--learning_rate', default=3e-4, type=float,
                             metavar='LR', help='初始学习率')
    optim_group.add_argument('--momentum', default=0.9, type=float,
                             help='SGD动量参数')
    optim_group.add_argument('--weight_decay', default=5e-4, type=float,
                             help='权重衰减系数')
    optim_group.add_argument('--nesterov', default=False, type=str2bool,
                             help='是否启用Nesterov动量 (仅SGD有效)')
    optim_group.add_argument('--clip_grad', default=1.0, type=float,
                    help='梯度裁剪阈值 (0表示禁用)')

    # -------------------- 学习率调度 --------------------
    scheduler_group = parser.add_argument_group('Scheduler Configuration')
    scheduler_group.add_argument('--scheduler', default='WarmupCosine',
                                 choices=['OneCycleLR','WarmupCosine','CosineAnnealingLR', 'ReduceLROnPlateau',
                                          'MultiStepLR', 'ConstantLR'],
                                 help='学习率调度策略')
    scheduler_group.add_argument('--warmup_epochs', type=int, default=20,
                                 help='学习率预热轮次 (默认: 10)')
    scheduler_group.add_argument('--max_lr_ratio', type=float, default=1.5,
                                 help='OneCycle最大学习率倍数')
    scheduler_group.add_argument('--min_lr', default=3e-5, type=float,
                                 help='最小学习率 (用于ReduceLROnPlateau)')
    scheduler_group.add_argument('--restart_cycle', default=100, type=int,
                                 help='重启周期')
    scheduler_group.add_argument('--factor', default=0.1, type=float,
                                 help='学习率衰减系数')
    scheduler_group.add_argument('--patience', default=2, type=int,
                                 help='耐心轮次 (用于ReduceLROnPlateau)')
    scheduler_group.add_argument('--milestones', default='1,2', type=str,
                                 help='里程碑轮次 (逗号分隔，用于MultiStepLR)')
    scheduler_group.add_argument('--gamma', default=2 / 3, type=float,
                                 help='学习率衰减系数 (用于MultiStepLR)')
    scheduler_group.add_argument('--early_stopping', default=-1, type=int, metavar='N',
                                 help='早停耐心轮次 (-1表示禁用)')

    return parser.parse_args()

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed()
    np.random.seed(42)
    random.seed(42)

def train(config, train_loader, model, criterion, optimizer, scheduler=None ,ema=None, epoch=0):
    """执行单个训练epoch"""

    # 初始化统计指标
    avg_meters = {
        'total_loss': AverageMeter(),  # 总损失
        'dice_loss': AverageMeter(),  # Dice损失
        'iou_loss': AverageMeter(),  # IoU损失
        'dice': AverageMeter(),  # Dice系数
        'iou': AverageMeter()  # IoU系数
    }

    model.train()
    scaler = GradScaler(
    init_scale=2 ** 16,
    growth_interval = 500,
    enabled = True
    )

    # 进度条配置
    with tqdm(total=len(train_loader), desc="\033[1;32mTraining\033[0m",
             bar_format="{l_bar}{bar:20}{r_bar}", dynamic_ncols=True) as pbar:
        for input, target, _ in train_loader:
            # 数据迁移到GPU
            input = input.to('cuda', non_blocking=True)
            target = target.to('cuda', non_blocking=True)

            # === 前向计算 ===
            with autocast(device_type='cuda', dtype=torch.float16):
                output = model(input)


                total_loss, loss_components = criterion(output, target, epoch)  # 接收损失分量
                # egeunet损失修改
                # if model.gt_ds and isinstance(output, tuple):
                #     deep_outputs, final_output = output
                #     total_loss, loss_components = criterion(final_output, target)
                #     metric_output = final_output
                # else:
                #     total_loss, loss_components = criterion(output, target)
                #     metric_output = output

            # === 反向传播 ===
            scaler.scale(total_loss).backward()  # 缩放梯度
            scaler.unscale_(optimizer)  # 解除梯度缩放
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config.get('clip_grad', 1.0),  # 从配置读取阈值
                norm_type=2
            )
            scaler.step(optimizer)  # 参数更新
            scaler.update()  # 缩放器状态更新
            optimizer.zero_grad(set_to_none=True)  # 梯度清零

            if ema is not None:
                ema.update()

            # 每个batch后更新学习率
            if scheduler and isinstance(scheduler, OneCycleLR):
                scheduler.step()

            if scheduler and isinstance(scheduler, WarmupCosineSchedule):
            # 每个epoch结束后更新学习率 (需在训练循环外调用)
                pass  # 这里保持空，实际在epoch循环外更新

                # === 指标计算 ===
            iou, dice = iou_score(output, target)

            # 更新统计指标
            total_loss, loss_components = criterion(output, target)
            avg_meters['total_loss'].update(total_loss.item(), input.size(0))
            avg_meters['dice_loss'].update(loss_components['dice'], input.size(0))
            avg_meters['iou_loss'].update(loss_components['iou'], input.size(0))
            avg_meters['dice'].update(dice, input.size(0))
            avg_meters['iou'].update(iou, input.size(0))

            # 更新进度信息
            pbar.set_postfix(OrderedDict(
                loss=f"{avg_meters['total_loss'].avg:.3f}",
                d_loss=f"{avg_meters['dice_loss'].avg:.2f}",
                i_loss=f"{avg_meters['iou_loss'].avg:.2f}",
                dice=f"{avg_meters['dice'].avg*100:.2f}",
                iou=f"{avg_meters['iou'].avg*100:.2f}"
            ))
            pbar.update(1)

    return OrderedDict(
        loss=avg_meters['total_loss'].avg,
        iou=avg_meters['iou'].avg
    )

def validate(config, val_loader, model, criterion, ema=None):
    """执行单个验证epoch"""

    # 初始化统计指标
    avg_meters = {
        'total_loss': AverageMeter(),  # 总损失
        'dice_loss': AverageMeter(),  # Dice损失
        'iou_loss': AverageMeter(),  # IoU损失
        'dice': AverageMeter(),  # Dice系数
        'iou': AverageMeter()  # IoU系数
    }

    # 评估模式设置
    model.eval()

    if ema is not None:
        ema.store()  # 备份原始参数
        ema.copy_to()  # 应用EMA参数

    with torch.no_grad():  # 禁用梯度计算
        with tqdm(val_loader, desc="\033[1;34mValidating\033[0m", bar_format="{l_bar}{bar:20}{r_bar}",
                 dynamic_ncols=True) as pbar:
            for input, target, _ in pbar:
                # 数据迁移到GPU
                input = input.to('cuda', non_blocking=True)
                target = target.to('cuda', non_blocking=True)

                # 模型推理
                output = model(input)

                # egeunet
                # if model.gt_ds and isinstance(output, tuple):
                #     deep_outputs, final_output = output
                #     total_loss, loss_components = criterion(final_output, target)
                #     iou, dice = iou_score(final_output, target)
                # else:
                #     total_loss, loss_components = criterion(output, target)
                #     iou, dice = iou_score(output, target)

                total_loss, loss_components = criterion(output, target)
                iou, dice = iou_score(output, target)

                avg_meters['total_loss'].update(total_loss.item(), input.size(0))
                avg_meters['dice_loss'].update(loss_components['dice'], input.size(0))
                avg_meters['iou_loss'].update(loss_components['iou'], input.size(0))
                avg_meters['dice'].update(dice, input.size(0))
                avg_meters['iou'].update(iou, input.size(0))

                # 更新进度信息
                pbar.set_postfix(OrderedDict(
                    loss=f"{avg_meters['total_loss'].avg:.3f}",
                    d_loss=f"{avg_meters['dice_loss'].avg:.2f}",
                    i_loss=f"{avg_meters['iou_loss'].avg:.2f}",
                    dice=f"{avg_meters['dice'].avg*100:.2f}",
                    iou=f"{avg_meters['iou'].avg*100:.2f}"
                ))

    if ema is not None:
        ema.restore()

    return OrderedDict(
        loss=avg_meters['total_loss'].avg,
        iou=avg_meters['iou'].avg,
        dice=avg_meters['dice'].avg
    )

def main():
    """主训练流程（完整五折交叉验证版本）"""
    # === 初始化配置 ===
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    seed_everything(42)
    config = vars(parse_args())

    # 实验名称生成规则
    if not config['name']:
        ds_tag = 'wDS' if config['deep_supervision'] else 'woDS'
        config['name'] = f"{config['dataset']}_{config['arch']}_{ds_tag}"
        # 新建层级目录：models/架构名/实验名/
        base_dir = os.path.join("models", config['arch'].lower(), config['name'])
        os.makedirs(base_dir, exist_ok=True)

        # 打印配置信息
    print('-' * 20 + ' 配置信息 ' + '-' * 20)
    [print(f"{k:15}: {v}") for k, v in config.items()]

    # === 数据加载 ===
    # 获取所有图像ID
    img_ids = [os.path.splitext(os.path.basename(p))[0]
               for p in glob(f"inputs/{config['dataset']}/images/*{config['img_ext']}")]

    # 固定测试集划分
    train_val_ids, test_img_ids = train_test_split(img_ids, test_size=0.2, random_state=42)
    test_ids_path = os.path.join('inputs', config['dataset'], 'test_ids.json')
    with open(test_ids_path, 'w') as f:
        json.dump(test_img_ids, f)

    # === 五折交叉验证 ===
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_ids)):
        print(f"\n\033[1;35m=== 开始第 {fold + 1}/5 折训练 ===\033[0m")

        # 创建fold专属输出目录
        fold_name = f"{config['name']}_fold{fold + 1}"
        # 更新路径为 models/架构名/实验名/fold名
        fold_dir = os.path.join("models", config['arch'].lower(), config['name'], fold_name)
        os.makedirs(fold_dir, exist_ok=True)  # 创建当前fold的目录

        # 配置文件路径同步更新
        config_path = os.path.join(fold_dir, "config.yml")
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        # 划分当前fold的数据
        train_img_ids = [train_val_ids[i] for i in train_idx]
        val_img_ids = [train_val_ids[i] for i in val_idx]

        # === 模型初始化 ===
        model = DPT(nclass=1).cuda()
        # model = UNetV2(n_classes=1, deep_supervision=False, pretrained_path=None).cuda()
        # model = LA_UNet().cuda()

        # model = MobileNetV3(model_mode="LARGE", num_classes=1).cuda()

        # model = EGEUNet(num_classes=1,
        #                 input_channels=3,
        #                 c_list=[8,16,24,32,48,64],
        #                 bridge=True,
        #                 gt_ds=True,
        #                 ).cuda()

        # model = UNext(num_classes=1, deep_supervision='false', input_channels=3).cuda()

        # model = UNet2Plus().cuda()

        # model = archs.__dict__['UKAN'](1, 3,False, embed_dims=[128, 160, 256],
        #                                        no_kan=True).cuda()

        # gpu_ids = [0]
        # model = MALUNet(num_classes=1,
        #                 input_channels=3,
        #                 c_list=[8, 16, 24, 32, 48, 64],
        #                 split_att='fc',
        #                 bridge=True)
        # model = torch.nn.DataParallel(model.cuda(), device_ids=gpu_ids, output_device=gpu_ids[0])

        # model = AttU_Net().cuda()

        # model = UNet(n_channels=3, n_classes=1, bilinear=False).cuda()

        ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
        ema.to('cuda')

        # === 损失函数 ===
        if config['loss'] == 'FocalBCEDiceLoss':
            criterion = losses.FocalBCEDiceLoss(
                alpha=0.75, gamma=2.0, smooth=1e-6,
                focal_weight=0.7, dice_weight=0.2, iou_weight=0.1
            ).cuda()
        else:
            criterion = (nn.BCEWithLogitsLoss() if config['loss'] == 'BCEWithLogitsLoss'
                         else losses.__dict__[config['loss']]()).cuda()

        # === 优化器配置 ===
        params = filter(lambda p: p.requires_grad, model.parameters())
        if config['optimizer'] == 'Adam':
            optimizer = optim.AdamW(params, lr=config['lr'],
                                    weight_decay=config['weight_decay'] * 0.5,
                                    betas=(0.9, 0.999), eps=1e-07)
        elif config['optimizer'] == 'SGD':
            optimizer = optim.SGD(params, lr=config['lr'],
                                  momentum=config['momentum'],
                                  nesterov=config['nesterov'],
                                  weight_decay=config['weight_decay'])

        # === 数据增强 ===
        train_transform = Compose([
            RandomRotate90(),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            Resize(config['input_h'], config['input_w']),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])
        val_transform = Compose([
            Resize(config['input_h'], config['input_w']),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])

        # === 数据集初始化 ===
        dataset_args = {
            'img_dir': f"inputs/{config['dataset']}/images",
            'mask_dir': f"inputs/{config['dataset']}/masks",
            'img_ext': config['img_ext'],
            'mask_ext': config['mask_ext']
        }
        train_dataset = Dataset(img_ids=train_img_ids,
                                transform=train_transform, **dataset_args)
        val_dataset = Dataset(img_ids=val_img_ids,
                              transform=val_transform, **dataset_args)

        # === 数据加载器 ===
        loader_args = {
            'batch_size': config['batch_size'],
            'num_workers': config['num_workers'],
            'pin_memory': True,
            'persistent_workers': True,
            'prefetch_factor': 2,
            'worker_init_fn': seed_worker,
            'generator': torch.Generator().manual_seed(42)
        }
        train_loader = DataLoader(train_dataset, shuffle=True,
                                  drop_last=True, **loader_args)
        val_loader = DataLoader(val_dataset, shuffle=False,
                                drop_last=False, **loader_args)

        # === 学习率调度 ===
        scheduler = None
        if config['scheduler'] == 'WarmupCosine':
            scheduler = WarmupCosineSchedule(
                optimizer,
                warmup_epochs=config['warmup_epochs'],
                total_epochs=config['epochs'],
                min_lr=config['min_lr'],
                restart_cycle=config['restart_cycle']
            )
        elif config['scheduler'] == 'OneCycleLR':
            scheduler = OneCycleLR(
                optimizer,
                max_lr=config['lr'] * 1.5,
                total_steps=config['epochs'] * len(train_loader),
                pct_start=0.4,
                anneal_strategy='linear',
                div_factor=5,
                final_div_factor=1e4,
                cycle_momentum=False
            )
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=config['factor'],
                patience=config['patience'], min_lr=config['min_lr'])
        elif config['scheduler'] == 'MultiStepLR':
            milestones = list(map(int, config['milestones'].split(',')))
            scheduler = lr_scheduler.MultiStepLR(optimizer,
                                                 milestones,
                                                 config['gamma'])

        # === 训练循环 ===
        log = OrderedDict([
            ('epoch', []), ('lr', []), ('loss', []), ('iou', []),
            ('val_loss', []), ('val_iou', []), ('val_dice', [])
        ])
        best_iou, best_dice, trigger = 0, 0, 0

        for epoch in range(config['epochs']):
            print(f'\n\033[1;36m=== Epoch [{epoch + 1}/{config["epochs"]}] ===\033[0m')

            # 深度监督调整
            if config['deep_supervision']:
                act_learn = 1 - math.cos(math.pi / 2 * epoch / config['epochs'])
                model.change_act(act_learn)

            # 训练与验证
            train_log = train(config, train_loader, model,
                              criterion, optimizer, scheduler, ema, epoch)
            val_log = validate(config, val_loader, model, criterion, ema)

            # 学习率调整
            if scheduler:
                if config['scheduler'] == 'ReduceLROnPlateau':
                    scheduler.step(val_log['loss'])
                elif isinstance(scheduler, (WarmupCosineSchedule)):
                    scheduler.step()

            # 日志记录
            log['epoch'].append(epoch)
            log['lr'].append(optimizer.param_groups[0]['lr'])
            log['loss'].append(train_log['loss'])
            log['iou'].append(train_log['iou'])
            log['val_loss'].append(val_log['loss'])
            log['val_iou'].append(val_log['iou'])
            log['val_dice'].append(val_log['dice'])

            # 模型保存
            if val_log['iou'] > best_iou:
                ema.store()
                ema.copy_to()
                model_dir = os.path.join("models", config['arch'].lower(), config['name'], fold_name)
                torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pth"))
                ema.restore()
                best_iou = val_log['iou']
                best_dice = val_log['dice']
                trigger = 0
                print(f"=> 保存最优模型 | IoU:{best_iou*100:.4f} Dice:{best_dice*100:.4f}")
            torch.save(model.state_dict(), os.path.join(model_dir, "last_model.pth"))

            # 早停机制
            trigger += 1
            if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
                print("=> 触发早停")
                break

            torch.cuda.empty_cache()

            # 保存实时日志
            log_path = os.path.join(
                "models",
                config['arch'].lower(),
                config['name'],
                fold_name,
                "log.csv"
            )
            pd.DataFrame(log).to_csv(log_path, index=False)

        # 记录当前fold结果
        fold_results.append({
            'fold': fold + 1,
            'best_iou': best_iou,
            'best_dice': best_dice,
            'final_epoch': epoch + 1
        })

    # === 交叉验证结果汇总 ===
    print("\n\033[1;35m=== 五折交叉验证结果汇总 ===\033[0m")
    df_results = pd.DataFrame(fold_results)
    summary = pd.DataFrame({
        'Metric': ['Mean IoU', 'Std IoU', 'Mean Dice', 'Std Dice'],
        'Value': [
            df_results['best_iou'].mean(),
            df_results['best_iou'].std(),
            df_results['best_dice'].mean(),
            df_results['best_dice'].std()
        ]
    })
    print(summary.round(4))
    summary_dir = os.path.join("models", config['arch'].lower(), config['name'])
    cv_summary_path = os.path.join(summary_dir, "cv_summary.csv")
    df_results.to_csv(cv_summary_path, index=False)


if __name__ == '__main__':
    main()

