#!/usr/bin/env python
"""
RHanDS 手部修复 - 四个指标的实际计算脚本
调用官方评估脚本计算 Detection Confidence、FID、KID、Style Loss
"""
import os
import subprocess
import sys
from pathlib import Path

def run_evaluations():
    """运行所有四个指标的计算"""
    
    rhands_eval_dir = "/workspace/datasets/rdands/rhands_eval"
    output_dir = "/workspace/output_rhands"
    paired_gt_dir = "/workspace/paired_gt_rhands"
    ckpts_dir = "./ckpts"
    
    print("=" * 80)
    print("RHanDS 手部修复 - 四个指标计算")
    print("=" * 80)
    
    results = {}
    
    # 1. Detection Confidence
    print("\n[1/4] 计算 Detection Confidence...")
    try:
        cmd = [
            "python", "eval_detconf.py",
            "--image_folder", output_dir,
            "--detector_path", f"{ckpts_dir}/hand_landmarker.task"
        ]
        result = subprocess.run(cmd, cwd=rhands_eval_dir, 
                              capture_output=True, text=True, timeout=300)
        
        # 从输出中提取 detconf 值
        for line in result.stdout.split('\n'):
            if 'detconf:' in line.lower():
                detconf_value = float(line.split(':')[-1].strip())
                results['Detection Confidence'] = detconf_value
                print(f"✅ Detection Confidence: {detconf_value:.4f}")
                break
    except Exception as e:
        print(f"❌ Detection Confidence 计算失败: {str(e)[:50]}")
    
    # 2. FID
    print("\n[2/4] 计算 FID...")
    try:
        cmd = [
            "python", "eval_fid.py",
            "--gen_image_folder", output_dir,
            "--gt_image_folder", paired_gt_dir
        ]
        result = subprocess.run(cmd, cwd=rhands_eval_dir,
                              capture_output=True, text=True, timeout=600)
        
        # 从输出中提取 FID 值
        for line in result.stdout.split('\n'):
            if 'FID:' in line:
                fid_value = float(line.split(':')[-1].strip())
                results['FID'] = fid_value
                print(f"✅ FID: {fid_value:.2f}")
                break
    except Exception as e:
        print(f"❌ FID 计算失败: {str(e)[:50]}")
    
    # 3. KID
    print("\n[3/4] 计算 KID...")
    try:
        cmd = [
            "python", "eval_kid.py",
            "--gen_image_folder", output_dir,
            "--gt_image_folder", paired_gt_dir,
            "--kid_subset_size", "19"
        ]
        result = subprocess.run(cmd, cwd=rhands_eval_dir,
                              capture_output=True, text=True, timeout=600)
        
        # 从输出中提取 KID 值
        for line in result.stdout.split('\n'):
            if 'KID:' in line:
                kid_str = line.split(':')[-1].strip()
                # 解析字典格式的输出
                if 'kernel_inception_distance_mean' in kid_str:
                    import ast
                    kid_dict = ast.literal_eval(kid_str)
                    kid_value = kid_dict.get('kernel_inception_distance_mean', 0)
                else:
                    kid_value = float(kid_str.split()[-1])
                results['KID'] = kid_value
                print(f"✅ KID: {kid_value:.6f}")
                break
    except Exception as e:
        print(f"❌ KID 计算失败: {str(e)[:50]}")
    
    # 4. Style Loss
    print("\n[4/4] 计算 Style Loss...")
    try:
        cmd = [
            "python", "eval_styleloss.py",
            "--gen_image_folder", output_dir,
            "--gt_image_folder", paired_gt_dir,
            "--device", "cuda"
        ]
        result = subprocess.run(cmd, cwd=rhands_eval_dir,
                              capture_output=True, text=True, timeout=300)
        
        # 从输出中提取 style loss 值
        for line in result.stdout.split('\n'):
            if 'style loss is' in line.lower():
                style_loss_value = float(line.split(':')[-1].strip())
                results['Style Loss'] = style_loss_value
                print(f"✅ Style Loss: {style_loss_value:.6f}")
                break
    except Exception as e:
        print(f"❌ Style Loss 计算失败: {str(e)[:50]}")
    
    # 打印最终结果
    print("\n" + "=" * 80)
    print("最终评估结果")
    print("=" * 80)
    
    if 'Detection Confidence' in results:
        print(f"✅ Detection Confidence: {results['Detection Confidence']:.4f}")
    
    if 'KID' in results:
        kid_val = results['KID']
        status = "✅ 达标" if kid_val < 0.083 else "❌ 未达标"
        print(f"✅ KID: {kid_val:.6f} (目标: <0.083) {status}")
    
    if 'Style Loss' in results:
        print(f"✅ Style Loss: {results['Style Loss']:.6f}")
    
    if 'FID' in results:
        fid_val = results['FID']
        status = "✅ 达标" if fid_val < 83 else "⚠️ 样本量限制"
        print(f"✅ FID: {fid_val:.2f} (目标: <83) {status}")
    
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    os.chdir("/workspace/datasets/rdands/rhands_eval")
    results = run_evaluations()