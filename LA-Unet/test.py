"""
医学图像分割测试脚本
"""
from unet import UNet
from attunet import AttU_Net
from malunet import MALUNet
from ukan import archs
from UNet2Plus.UNet2Plus import UNet2Plus
from unext.model import UNext
from egeunet.egeunet import EGEUNet
from v3.mobilenetv3 import  MobileNetV3
#from lvunet.LV_UNet import LV_UNet
from UNet_v2.UNet_v2 import UNetV2
# -------------------- 基础库导入 --------------------
import os
import argparse
import random
from glob import glob
import json
import cv2
import numpy as np
import pandas as pd
import torch
import yaml
import torch.backends.cudnn as cudnn
from albumentations import Compose, Resize
from albumentations.augmentations import transforms
from sklearn.model_selection import train_test_split
from skimage import morphology
from skimage.morphology import remove_small_holes 
from thop import profile
from torch.utils.data import DataLoader
from tqdm import tqdm


# -------------------- 自定义模块导入 --------------------
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter, str2bool
from LA_UNet import LA_UNet

# -------------------- 环境配置 --------------------
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'  # 禁用Albumentations自动更新
os.environ['ALBUMENTATIONS_NO_UPDATE_CHECK'] = '1'
cudnn.benchmark = True  # 启用CUDA加速

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='医学图像分割训练参数')

    # 模型配置参数
    model_group = parser.add_argument_group('模型配置')
    model_group.add_argument('--arch', default='UNetV2', help='模型架构名称（默认: unet）')
    model_group.add_argument('--name', default='busi_UNetV2_woDS',
                             help='模型保存名称（默认: isic_LV_UNet_woDS）,'
                                  'busi_LV_UNet_woDS,'
                                  'cvc_clinicdb_LV_UNet_woDS,'
                                  'CVC_ColonDB_LV_UNet_woDS,'
                                  'Kvasir_SEG_LV_UNet_woDS')
    model_group.add_argument('--deploy', default=False, type=str2bool,
                             help='是否使用部署模式（默认: False）')
    model_group.add_argument('--save_outputs', action='store_true', default=True, help='保存预测结果')

    model_group.add_argument('--input_w', default=256, type=int,
                             help='输入图像宽度')
    model_group.add_argument('--input_h', default=256, type=int,
                             help='输入图像高度')


    return parser.parse_args()
def seed_worker(worker_id):
    worker_seed = torch.initial_seed()
    np.random.seed(42)
    random.seed(42)

class AverageMeter:
    """计算并存储平均值和当前值"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def main():
    """模型验证主流程（五折交叉验证版本）"""
    args = parse_args()

    # === 参数解析 ===
    model_arch = args.arch.lower()  # 模型架构名称
    exp_name = args.name  # 实验名称
    save_outputs = args.save_outputs  # 是否保存预测结果
    deploy_mode = args.deploy  

    # === 加载测试集划分 ===
    temp_config_path = f"models/{model_arch}/{exp_name}/{exp_name}_fold1/config.yml"
    with open(temp_config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    _, test_img_ids = train_test_split(img_ids, test_size=0.8, random_state=42)
    test_img_ids, _ = train_test_split(test_img_ids, test_size=0.2, random_state=42)

    # === 五折测试循环 ===
    fold_results = []
    for fold in range(1, 6):
        fold_name = f"{exp_name}_fold{fold}"
        print(f"\n\033[1;35m=== 测试第 {fold}/5 折 ===\033[0m")

        # === 加载当前fold配置 ===

        with open(f'models/{model_arch}/{exp_name}/{fold_name}/config.yml') as f:
            config = yaml.safe_load(f)

        # === 模型初始化 ===

        model = LA_UNet().cuda()



        # === 加载权重 ===
        weight_path = f'models/{model_arch}/{exp_name}/{fold_name}/best_model.pth'
        model.load_state_dict(torch.load(weight_path, weights_only=True))

        # 部署模式切换
        if deploy_mode:
            model.switch_to_deploy()
        model.eval()

        # === 数据预处理 ===
        val_transform = Compose([
            Resize(config['input_h'], config['input_w']),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])

        # === 数据集初始化 ===
        val_dataset = Dataset(
            img_ids=test_img_ids,
            img_dir=f"inputs/{config['dataset']}/images",
            mask_dir=f"inputs/{config['dataset']}/masks",
            img_ext=config['img_ext'],
            mask_ext=config['mask_ext'],
            transform=val_transform
        )

        # === 数据加载器 ===
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            pin_memory=True,
            persistent_workers=True,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(42)
        )

        # === 指标初始化 ===
        iou_meter = AverageMeter()
        dice_meter = AverageMeter()

        # === 创建输出目录 ===
        # 修改点4：适配新的输出路径结构
        if save_outputs:
            # outputs目录结构: outputs/{model_arch}/{exp_name}/{fold_name}/
            output_root = os.path.join('outputs', model_arch, exp_name)
            output_dir = os.path.join(output_root, fold_name)

            # 递归创建多级目录
            os.makedirs(output_dir, exist_ok=True)

            # 创建类别子目录
            num_classes = config.get('num_classes', 1)
            for c in range(num_classes):
                class_dir = os.path.join(output_dir, str(c))
                os.makedirs(class_dir, exist_ok=True)

        # === 推理循环 ===
        with torch.no_grad():
            for inputs, targets, meta in tqdm(val_loader, desc='推理进度'):
                # 数据转移
                inputs = inputs.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

                # 前向传播
                outputs = model(inputs)


                # egeunet
                # if isinstance(outputs, tuple) or isinstance(outputs, list):
                #     main_output = outputs[-1]
                # else:
                #     main_output = outputs
                # iou, dice = iou_score(main_output, targets)
                # iou_meter.update(iou, inputs.size(0))
                # dice_meter.update(dice, inputs.size(0))

                # 指标计算
                iou, dice = iou_score(outputs, targets)
                iou_meter.update(iou, inputs.size(0))
                dice_meter.update(dice, inputs.size(0))

                # 结果保存
                if save_outputs:
                    outputs = torch.sigmoid(outputs).cpu().numpy()
                    outputs = (outputs >= 0.5).astype(np.uint8)
                # if save_outputs:
                #     outputs = torch.sigmoid(main_output).cpu().numpy()
                #     outputs = (outputs >= 0.5).astype(np.uint8)

                    for i in range(outputs.shape[0]):
                        img_id = meta['img_id'][i]
                        for c in range(num_classes):
                            cv2.imwrite(
                                # 修改点5：适配新的输出文件路径
                                os.path.join(output_dir, str(c), f"{img_id}.png"),
                                (outputs[i, c] * 255).astype(np.uint8)
                            )

        # === 记录当前fold结果 ===
        fold_results.append({
            'fold': fold,
            'iou': iou_meter.avg.item(),
            'dice': dice_meter.avg.item()
        })
        print(f"Fold {fold} 结果: IoU {iou_meter.avg * 100:.4f} | Dice {dice_meter.avg * 100:.4f}")

        # === 清理显存 ===
        torch.cuda.empty_cache()

    # === 结果汇总 ===
    print("\n\033[1;35m=== 最终测试结果 ===\033[0m")
    df_results = pd.DataFrame(fold_results)
    summary = pd.DataFrame({
        'Metric': ['Mean IoU', 'Std IoU', 'Mean Dice', 'Std Dice'],
        'Value': [
            df_results['iou'].mean(),
            df_results['iou'].std(),
            df_results['dice'].mean(),
            df_results['dice'].std()
        ]
    })

    # 打印详细结果
    print("\n\033[1;36m各折详细结果:\033[0m")
    print(df_results.round(4))
    print("\n\033[1;36m汇总统计:\033[0m")
    print(summary.round(4))

    # 保存结果
    output_root = os.path.join('outputs', model_arch, exp_name)
    os.makedirs(output_root, exist_ok=True)
    summary_path = os.path.join(output_root, 'test_summary.csv')
    df_results.to_csv(summary_path, index=False)

if __name__ == '__main__':
    main()
