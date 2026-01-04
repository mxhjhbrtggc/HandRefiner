"""
第 2 步：批量修复图像
"""

import os
import time
import subprocess
from pathlib import Path

print("="*70)
print("步骤 2: 批量修复图像")
print("="*70)

input_dir = '/workspace/datasets/FreiHAND_512x512_100'
output_dir = '/workspace/output_freihand_pipeline'
os.makedirs(output_dir, exist_ok=True)

prompt = "a person with hand gesture, high quality, clear image"

images = sorted(list(Path(input_dir).glob("*.jpg")))
print(f"\n✅ 准备修复 {len(images)} 张图像\n")

success_count = 0
failed_count = 0
start_time = time.time()

for idx, img_path in enumerate(images, 1):
    img_name = img_path.stem
    output_file = Path(output_dir) / f"{img_name}_0.jpg"
    
    print(f"[{idx}/{len(images)}] {img_name}...", end=" ", flush=True)
    
    try:
        if output_file.exists():
            print("(已存在) ✅")
            success_count += 1
            continue
        
        cmd = [
            'python', '/workspace/handrefiner.py',
            '--input_img', str(img_path),
            '--out_dir', output_dir,
            '--strength', '0.55',
            '--prompt', prompt,
            '--finetuned', 'False',
            '--seed', '1'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0 and output_file.exists():
            size_mb = output_file.stat().st_size / (1024**2)
            print(f"✅ ({size_mb:.1f}MB)")
            success_count += 1
        else:
            print("❌")
            failed_count += 1
            
    except subprocess.TimeoutExpired:
        print("❌ (超时)")
        failed_count += 1
    except Exception as e:
        print(f"❌ ({str(e)[:30]})")
        failed_count += 1

elapsed = time.time() - start_time

print(f"\n{'='*70}")
print(f"📊 修复统计：")
print(f"   ✅ 成功：{success_count} 张")
print(f"   ❌ 失败：{failed_count} 张")
print(f"   ⏱️ 总耗时：{elapsed/60:.1f} 分钟")
if success_count > 0:
    print(f"   ⏱️ 平均：{elapsed/success_count:.1f} 秒/张")
print(f"{'='*70}")
