"""
第 1 步：预处理图像到 512x512
"""

from pathlib import Path
from PIL import Image

print("="*70)
print("步骤 1: 预处理图像到 512x512")
print("="*70)

input_dir = '/workspace/datasets/FreiHAND/training/rgb'
output_dir = '/workspace/datasets/FreiHAND_512x512_100'
max_images = 100

input_path = Path(input_dir)
output_path = Path(output_dir)
output_path.mkdir(exist_ok=True, parents=True)

# 获取前 100 张图像
images = sorted(list(input_path.glob("*.jpg")))[:max_images]
print(f"\n✅ 找到 {len(images)} 张图像\n")

count = 0
for idx, img_file in enumerate(images, 1):
    output_file = output_path / img_file.name
    
    # 跳过已处理的
    if output_file.exists():
        count += 1
        continue
    
    print(f"[{idx}/{len(images)}] {img_file.name}...", end=" ", flush=True)
    try:
        img = Image.open(img_file).convert('RGB')
        img_resized = img.resize((512, 512), Image.LANCZOS)
        img_resized.save(output_file, quality=95)
        count += 1
        print("✅")
    except Exception as e:
        print(f"❌ ({str(e)[:30]})")

print(f"\n{'='*70}")
print(f"✅ 预处理完成：{count}/{len(images)} 张")
print(f"{'='*70}")
