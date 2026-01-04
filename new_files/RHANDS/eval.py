#!/usr/bin/env python
"""
RHanDS 图像配对脚本
从修复的图像中提取对应的GT图像
"""
import os
import shutil
from pathlib import Path


def extract_image_id(filename):
    """
    从文件名提取ID
    修复后: acrobatics_000000000000_0_0.jpg -> acrobatics_000000000000_0
    """
    basename = os.path.splitext(os.path.basename(filename))[0]
    # 去掉最后的 _0
    return basename.rsplit('_', 1)[0]


def create_paired_gt_dataset():
    """
    为已修复的图像创建对应的GT图像数据集
    """
    # 路径配置
    gen_folder = "/workspace/output_rhands"
    gt_folder = "/workspace/datasets/rdands/rhands_multistyle_hand/gthand"
    
    # 输出文件夹
    paired_gt_folder = "/workspace/paired_gt_rhands"
    
    os.makedirs(paired_gt_folder, exist_ok=True)
    
    # 扫描已修复的图像
    gen_files = sorted([f for f in os.listdir(gen_folder) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    print("=" * 80)
    print("RHanDS GT 图像配对")
    print("=" * 80)
    print(f"\n已修复图像: {len(gen_files)} 张\n")
    
    matched_count = 0
    not_found_count = 0
    
    # 构建GT文件映射字典（提高查询速度）
    gt_dict = {}
    for f in os.listdir(gt_folder):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            gt_id = os.path.splitext(f)[0]
            gt_dict[gt_id] = f
    
    for idx, gen_file in enumerate(gen_files, 1):
        # 提取ID
        gen_id = extract_image_id(gen_file)
        
        # 在GT中找对应文件
        gt_file = gt_dict.get(gen_id)
        
        if gt_file:
            # 复制文件
            src_gt = os.path.join(gt_folder, gt_file)
            dst_gt = os.path.join(paired_gt_folder, gt_file)
            
            shutil.copy2(src_gt, dst_gt)
            
            print(f"[{idx:2d}] ✅ {gen_id}")
            print(f"      GT: {gt_file}")
            matched_count += 1
        else:
            print(f"[{idx:2d}] ❌ {gen_id} - 未找到GT文件")
            not_found_count += 1
    
    print("\n" + "=" * 80)
    print(f"✅ 配对成功: {matched_count} 张")
    print(f"❌ 配对失败: {not_found_count} 张")
    print(f"\n配对结果:")
    print(f"  修复图像: {gen_folder}")
    print(f"  GT图像: {paired_gt_folder}")
    print("\n下一步，运行官方评估脚本:")
    print(f"  cd /workspace/rhands_eval/rhands_eval")
    print(f"  python eval_fid.py \\")
    print(f"    --gen_image_folder {gen_folder} \\")
    print(f"    --gt_image_folder {paired_gt_folder}")
    print("=" * 80)
    
    return matched_count


if __name__ == "__main__":
    create_paired_gt_dataset()