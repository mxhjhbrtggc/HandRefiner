import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_fid_given_paths
import os

def load_image(img_path):
    """加载图像"""
    img = Image.open(img_path).convert('RGB')
    return img

def calculate_lpips_score(img1_path, img2_path):
    """计算 LPIPS 相似度（范围 0-1，越小越相似）"""
    import lpips
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = lpips.LPIPS(net='alex').to(device)
    
    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')
    
    # 转换为张量 (1, 3, H, W)，值范围 [-1, 1]
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    img1_t = transform(img1).unsqueeze(0).to(device)
    img2_t = transform(img2).unsqueeze(0).to(device)
    
    with torch.no_grad():
        lpips_score = loss_fn(img1_t, img2_t).item()
    
    return lpips_score

def calculate_mse(img1_path, img2_path):
    """计算 MSE (Mean Squared Error)"""
    img1 = np.array(Image.open(img1_path).convert('RGB'), dtype=np.float32) / 255.0
    img2 = np.array(Image.open(img2_path).convert('RGB'), dtype=np.float32) / 255.0
    
    # 调整到相同大小
    h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
    img1 = img1[:h, :w]
    img2 = img2[:h, :w]
    
    mse = np.mean((img1 - img2) ** 2)
    psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
    
    return mse, psnr

def calculate_ssim(img1_path, img2_path):
    """计算 SSIM (Structural Similarity)"""
    from skimage.metrics import structural_similarity as ssim
    
    img1 = np.array(Image.open(img1_path).convert('RGB'), dtype=np.float32)
    img2 = np.array(Image.open(img2_path).convert('RGB'), dtype=np.float32)
    
    # 调整到相同大小
    h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
    img1 = img1[:h, :w]
    img2 = img2[:h, :w]
    
    ssim_score = ssim(img1, img2, channel_axis=2, data_range=255)
    
    return ssim_score

if __name__ == '__main__':
    input_img = 'test/1.jpg'
    output_img = 'output/1_0.jpg'
    
    print("=" * 60)
    print("HandRefiner 修复质量评估")
    print("=" * 60)
    print(f"输入图像：{input_img}")
    print(f"输出图像：{output_img}")
    print()
    
    # 检查文件是否存在
    if not os.path.exists(input_img):
        print(f"错误：找不到输入图像 {input_img}")
        exit(1)
    
    if not os.path.exists(output_img):
        print(f"错误：找不到输出图像 {output_img}")
        exit(1)
    
    print("计算中...")
    print()
    
    # 1. MSE 和 PSNR
    print("1️⃣ MSE & PSNR (像素级相似度)")
    print("-" * 60)
    try:
        mse, psnr = calculate_mse(input_img, output_img)
        print(f"   MSE  (越小越好)：{mse:.6f}")
        print(f"   PSNR (越大越好)：{psnr:.2f} dB")
        print()
    except Exception as e:
        print(f"   计算失败：{e}")
        print()
    
    # 2. SSIM
    print("2️⃣ SSIM (结构相似度)")
    print("-" * 60)
    try:
        ssim_score = calculate_ssim(input_img, output_img)
        print(f"   SSIM (范围 0-1，越接近 1 越好)：{ssim_score:.4f}")
        if ssim_score > 0.8:
            print(f"   评价：✅ 非常相似（修复保留了原始结构）")
        elif ssim_score > 0.6:
            print(f"   评价：👍 较相似（有适度修改）")
        else:
            print(f"   评价：⚠️ 差异较大（进行了显著修改）")
        print()
    except Exception as e:
        print(f"   计算失败：{e}")
        print()
    
    # 3. LPIPS
    print("3️⃣ LPIPS (感知相似度)")
    print("-" * 60)
    try:
        lpips_score = calculate_lpips_score(input_img, output_img)
        print(f"   LPIPS (范围 0-1，越小越相似)：{lpips_score:.4f}")
        if lpips_score < 0.1:
            print(f"   评价：✅ 非常相似（几乎无感知差异）")
        elif lpips_score < 0.3:
            print(f"   评价：👍 相似（小幅修改）")
        else:
            print(f"   评价：⚠️ 差异较大（明显修改）")
        print()
    except Exception as e:
        print(f"   计算失败：{e}")
        print()
    
    print("=" * 60)
    print("评估完成！")
    print("=" * 60)

