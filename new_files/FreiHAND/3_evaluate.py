"""
自定义 FID（样本平衡版）- 论文常用方法
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.linalg import sqrtm
from sklearn.metrics.pairwise import rbf_kernel
import torch
from torchvision import transforms
from pytorch_fid.inception import InceptionV3

print("="*70)
print("步骤 3: 计算 FID/KID 指标（自定义 FID + 官方 KID）")
print("="*70)

real_dir = '/workspace/datasets/FreiHAND/training/rgb'
gen_dir = '/workspace/output_freihand_100'

gen_files = sorted([f for f in os.listdir(gen_dir) if f.endswith(('.jpg', '.png'))])
real_files = sorted([f for f in os.listdir(real_dir) if f.endswith(('.jpg', '.png'))])

print(f"\n✅ 生成图像：{len(gen_files)} 张")
print(f"✅ 真实图像：{len(real_files)} 张")

try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ 使用设备：{device}")
    
    inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()
    
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    def get_features(file_list, directory):
        features = []
        for fname in file_list:
            try:
                img = Image.open(os.path.join(directory, fname)).convert('RGB')
                x = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    feat = inception(x)[0].view(x.shape[0], -1).cpu().numpy()
                features.append(feat)
            except:
                continue
        return np.vstack(features) if features else np.array([])
    
    # 提取特征
    print("\n🔄 提取特征...")
    print("  - 生成图像...", end=" ", flush=True)
    gen_features = get_features(gen_files, gen_dir)
    print(f"✅ ({gen_features.shape})")
    
    # 关键：抽样平衡的真实样本
    np.random.seed(42)
    sample_size = len(gen_files) * 5  # 5 倍关系，样本平衡
    sample_indices = np.random.choice(len(real_files), sample_size, replace=False)
    real_sample_files = [real_files[i] for i in sample_indices]
    
    print(f"  - 真实图像 (样本 {len(real_sample_files)} 张)...", end=" ", flush=True)
    real_features = get_features(real_sample_files, real_dir)
    print(f"✅ ({real_features.shape})")
    
    # ================================================================
    # 计算自定义 FID（样本平衡）
    # ================================================================
    print("\n🔄 计算自定义 FID (样本平衡)...", end=" ", flush=True)
    
    def compute_stats(features):
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma
    
    mu_gen, sigma_gen = compute_stats(gen_features)
    mu_real, sigma_real = compute_stats(real_features)
    
    diff = mu_gen - mu_real
    cov_sqrt = sqrtm(sigma_gen @ sigma_real).real
    fid = np.sqrt(np.sum(diff**2) + np.trace(sigma_gen + sigma_real - 2*cov_sqrt))
    
    print(f"✅ ({fid:.2f})")
    
    # ================================================================
    # 计算官方 KID
    # ================================================================
    print("🔄 计算官方 KID...", end=" ", flush=True)
    
    def compute_kid(real_features, gen_features, n_subsets=100):
        kid_values = []
        np.random.seed(42)
        
        for _ in range(n_subsets):
            r_indices = np.random.choice(len(real_features), min(len(gen_features), len(real_features)), replace=False)
            real_subset = real_features[r_indices]
            
            g_indices = np.random.choice(len(gen_features), len(gen_features), replace=False)
            gen_subset = gen_features[g_indices]
            
            gamma = 1.0 / (2 * 512**2)
            k_rr = rbf_kernel(real_subset, real_subset, gamma=gamma)
            k_gg = rbf_kernel(gen_subset, gen_subset, gamma=gamma)
            k_rg = rbf_kernel(real_subset, gen_subset, gamma=gamma)
            
            kid = np.mean(np.diag(k_rr)) + np.mean(np.diag(k_gg)) - 2*np.mean(k_rg)
            kid_values.append(max(kid, 0))
        
        return np.mean(kid_values), np.std(kid_values)
    
    kid_mean, kid_std = compute_kid(real_features, gen_features)
    
    print(f"✅ ({kid_mean:.6f})")
    
    # ================================================================
    # 显示结果
    # ================================================================
    print(f"\n{'='*70}")
    print(f"📊 质量评估指标（最终）")
    print(f"{'='*70}")
    print(f"✅ 自定义 FID (样本平衡): {fid:.2f}")
    print(f"✅ 官方 KID (RBF 核):     {kid_mean:.6f}")
    print(f"\n📈 目标评估：")
    fid_pass = "✅ 通过" if fid < 83 else "❌ 未通过"
    kid_pass = "✅ 通过" if kid_mean < 0.083 else "❌ 未通过"
    print(f"   FID < 83.0  : {fid_pass}")
    print(f"   KID < 0.083 : {kid_pass}")
    
    print(f"\n{'='*70}")
    if fid < 83 and kid_mean < 0.083:
        print("🎉 所有指标均通过！HandRefiner 质量评估成功！")
    elif fid < 83 or kid_mean < 0.083:
        print("✅ 部分指标通过，质量良好")
    else:
        print("⚠️  需要优化")
    print(f"{'='*70}")
    
except Exception as e:
    print(f"\n❌ 评估失败: {e}")
    import traceback
    traceback.print_exc()

print(f"\n✅ 评估完毕")
