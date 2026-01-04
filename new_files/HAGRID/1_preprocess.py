"""
第 1 步：预处理 HAGRID no_gesture 图像到 512x512
"""

from pathlib import Path
from PIL import Image

print("="*70)
print("步骤 1: 预处理 HAGRID no_gesture 图像到 512x512")
print("="*70)

input_dir = '/workspace/datasets/HAGRID'
output_dir = '/workspace/datasets/HAGRID_512x512'

input_path = Path(input_dir)
if not input_path.exists():
    print(f"\n❌ 输入目录不存在: {input_dir}")
    exit(1)

output_path = Path(output_dir)
output_path.mkdir(exist_ok=True, parents=True)

# 获取前 100 张图像
images = sorted(list(input_path.glob("*.jpg")))[:100]
print(f"\n✅ 找到 {len(images)} 张图像（预处理前 100 张）\n")

if len(images) == 0:
    print(f"❌ 未找到 JPG 文件，请检查目录: {input_dir}")
    exit(1)

count = 0
for idx, img_file in enumerate(images, 1):
    output_file = output_path / img_file.name
    
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
