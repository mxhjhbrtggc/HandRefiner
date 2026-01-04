#!/usr/bin/env python
"""
RHanDS 手部图像修复脚本（排序版）
修复前50张畸形手部图像 - 按文件名排序
自动跳过已修复和无法修复的图片
"""
import os
import subprocess
import sys
from pathlib import Path

# 云端配置
MALFORMED_HAND_DIR = "/workspace/datasets/rdands/rhands_multistyle_hand/malformed_hand"
HANDREFINER_SCRIPT = "/workspace/handrefiner.py"
OUTPUT_DIR = "/workspace/output_rhands"
WORKSPACE_DIR = "/workspace"
FAILED_FILE = os.path.join(OUTPUT_DIR, ".failed_list.txt")
MAX_IMAGES = 50

def load_failed_list():
    """加载失败列表"""
    if os.path.exists(FAILED_FILE):
        with open(FAILED_FILE, 'r') as f:
            return set(line.strip() for line in f if line.strip())
    return set()

def save_failed_list(failed_set):
    """保存失败列表"""
    with open(FAILED_FILE, 'w') as f:
        for filename in sorted(failed_set):
            f.write(f"{filename}\n")

def extract_image_id(filename):
    """
    从文件名提取图像ID用于匹配
    输入: acrobatics_000000000368_1.jpg
    输出: acrobatics_000000000368_1
    """
    return os.path.splitext(filename)[0]

def load_repaired_ids():
    """加载已修复的文件ID（从输出目录）"""
    repaired_ids = set()
    if os.path.exists(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                # 输出格式: acrobatics_000000000716_0_0.jpg
                # 提取ID: acrobatics_000000000716_0
                parts = os.path.splitext(f)[0].rsplit('_', 1)[0]
                repaired_ids.add(parts)
    return repaired_ids

def run_repair(idx, filename, input_path, output_dir, seed):
    """运行单个修复任务"""
    
    cmd = [
        "python",
        HANDREFINER_SCRIPT,
        "--input_img", input_path,
        "--out_dir", output_dir,
        "--strength", "0.55",
        "--prompt", "a good quality hand, realistic",
        "--seed", str(seed),
        "--finetuned", "False",
        "--num_samples", "1"
    ]
    
    print(f"[{idx:2d}/50] 🔄 {filename}...", end=" ", flush=True)
    
    try:
        result = subprocess.run(
            cmd,
            cwd=WORKSPACE_DIR,
            timeout=300,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅")
            return True
        else:
            print("❌")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏱️ (超时)")
        return False
        
    except Exception as e:
        print("❌")
        return False

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(HANDREFINER_SCRIPT):
        print(f"❌ 脚本不存在: {HANDREFINER_SCRIPT}")
        return 1
    
    if not os.path.exists(MALFORMED_HAND_DIR):
        print(f"❌ 目录不存在: {MALFORMED_HAND_DIR}")
        return 1
    
    # 加载已修复的文件ID和失败列表
    repaired_ids = load_repaired_ids()
    failed_set = load_failed_list()
    
    # 获取所有图像并按文件名排序
    all_files = sorted([f for f in os.listdir(MALFORMED_HAND_DIR) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    print("=" * 80)
    print("RHanDS 手部图像修复 - 前 50 张（已排序）")
    print("=" * 80)
    print(f"找到 {len(all_files)} 张")
    print(f"已修复: {len(repaired_ids)} 张")
    print(f"曾失败: {len(failed_set)} 张\n")
    
    success_count = 0
    failed_count = 0
    already_repaired_count = 0
    already_failed_count = 0
    idx = 0
    
    for filename in all_files[:MAX_IMAGES]:
        idx += 1
        image_id = extract_image_id(filename)
        input_path = os.path.join(MALFORMED_HAND_DIR, filename)
        seed = 42 + idx
        
        # 检查是否已修复（基于ID匹配）
        if image_id in repaired_ids:
            print(f"[{idx:2d}/50] ⏭️  {filename}... ⏭️ (已修复)")
            already_repaired_count += 1
            continue
        
        # 检查是否之前失败过 - 直接跳过，不重新运行
        if filename in failed_set:
            print(f"[{idx:2d}/50] ⏭️  {filename}... ⏭️ (曾失败)")
            already_failed_count += 1
            continue
        
        if run_repair(idx, filename, input_path, OUTPUT_DIR, seed):
            success_count += 1
        else:
            failed_count += 1
            failed_set.add(filename)
            save_failed_list(failed_set)
    
    output_files = [f for f in os.listdir(OUTPUT_DIR) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print("\n" + "=" * 80)
    print(f"✅ 本次新增成功: {success_count} 张")
    print(f"❌ 本次新增失败: {failed_count} 张")
    print(f"📝 本次已修复跳过: {already_repaired_count} 张")
    print(f"📝 本次曾失败跳过: {already_failed_count} 张")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"📊 前50张处理总计: {already_repaired_count + already_failed_count + success_count + failed_count} 张")
    print(f"📊 输出目录总计: {len(output_files)} 张")
    print(f"📄 失败列表: {FAILED_FILE}")
    print("=" * 80)
    
    return 0 if success_count > 0 else 1

if __name__ == "__main__":
    sys.exit(main())