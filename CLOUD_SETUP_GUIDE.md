# Cloud Studio 云端完整配置指南

## 📌 概述

本指南包含在 Cloud Studio GPU 环境中完整配置 HandRefiner 的所有步骤。

**预计耗时**: 30-45 分钟（主要时间用于下载模型）

---

## 🚀 第一步：基础环境配置

### 1.1 克隆项目代码

```bash
# 进入主目录
cd ~

# 克隆你的 GitHub 仓库
git clone https://github.com/mxhjhbrtggc/HandRefiner.git
cd HandRefiner

# 验证代码
git log --oneline -3
```

**预期输出**:
```
d2cbc1c Initial commit: Add HandRefiner project with updated .gitignore
f07e196 Update README.md
eeaae95 Update README.md
```

---

### 1.2 创建虚拟环境

```bash
# 创建虚拟环境
python3 -m venv venv_gpu

# 激活虚拟环境
source venv_gpu/bin/activate

# 验证
python --version  # 应显示 Python 3.8+
pip --version
```

---

### 1.3 安装核心依赖

```bash
# 升级 pip
pip install --upgrade pip

# 安装 PyTorch（CUDA 支持）
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu118

# 验证 CUDA
python -c "import torch; print(f'PyTorch 版本: {torch.__version__}'); print(f'CUDA 可用: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"无\"}')"
```

**预期输出**:
```
PyTorch 版本: 2.0.0+cu118
CUDA 可用: True
GPU: NVIDIA A100 (或 V100/RTX 等)
```

---

### 1.4 安装 HandRefiner 依赖

```bash
# 安装所有依赖
pip install -r requirements.txt

# 验证主要库
python -c "import cv2, numpy, albumentations, mediapipe, transformers; print('✅ 所有依赖安装成功')"
```

---

## 🔗 第二步：安装 MeshGraphormer

这是最关键的步骤，用于手部网格重建。

### 2.1 克隆 MeshGraphormer

```bash
# 确保在 HandRefiner 目录中
pwd  # 应显示 .../HandRefiner

# 克隆 MeshGraphormer
git clone --recursive https://github.com/microsoft/MeshGraphormer.git
cd MeshGraphormer

# 显示目录结构
ls -la
```

### 2.2 安装 MeshGraphormer 依赖

```bash
# 安装 manopth（MANO 库）
pip install ./manopth/.

# 验证安装
python -c "import manopth; print('✅ manopth 安装成功')"
```

### 2.3 创建模型文件夹

```bash
# 确保在 MeshGraphormer 目录中
pwd  # 应显示 .../MeshGraphormer

# 创建模型文件夹
mkdir -p models
mkdir -p src/modeling/data

# 显示文件夹结构
tree models src -L 2  # 或使用 ls -R models src
```

### 2.4 复制脚本和配置

```bash
# 复制关键脚本文件（假设当前在 MeshGraphormer 目录）
cp ../scripts/download_models.sh scripts/download_models.sh
cp ../scripts/_gcnn.py src/modeling/_gcnn.py
cp ../scripts/_mano.py src/modeling/_mano.py
cp ../scripts/config.py src/modeling/data/config.py

# 验证文件是否复制成功
ls -la scripts/download_models.sh
ls -la src/modeling/_gcnn.py
ls -la src/modeling/_mano.py
ls -la src/modeling/data/config.py
```

### 2.5 下载预训练模型

```bash
# 确保在 MeshGraphormer 目录
cd ~/HandRefiner/MeshGraphormer

# 下载 GraphOrmer 和 HRNet 权重
bash scripts/download_models.sh

# 验证下载
ls -lh models/graphormer_release/
ls -lh models/hrnet/
```

**预期文件**:
```
models/graphormer_release/
├── graphormer_hand_state_dict.bin  (~200MB)

models/hrnet/
├── hrnetv2_w64_imagenet_pretrained.pth  (~180MB)
├── cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml
```

### 2.6 手动下载 MANO 模型（关键步骤）

⚠️ **MANO 模型需要手动下载**（自动下载需要注册）

```bash
# 方法 1：如果已经下载到本地，从本地上传
# （在本地执行）
scp ~/MANO_RIGHT.pkl user@cloud-studio:/home/user/HandRefiner/MeshGraphormer/src/modeling/data/

# 方法 2：在云端直接下载（如果您有访问权限）
cd ~/HandRefiner/MeshGraphormer/src/modeling/data/
wget -O MANO_RIGHT.pkl "https://your-mano-download-link"  # 替换实际链接

# 方法 3：通过浏览器手动下载
# 1. 访问 https://mano.is.tue.mpg.de/
# 2. 注册并登录
# 3. 下载 MANO_RIGHT.pkl
# 4. 在 Cloud Studio 中上传到 src/modeling/data/

# 验证文件
ls -lh src/modeling/data/MANO_RIGHT.pkl  # 应显示 ~400MB
```

### 2.7 验证 MeshGraphormer 安装

```bash
# 测试导入
python -c "
import sys
sys.path.insert(0, 'src')
from modeling import MeshGraphormer
print('✅ MeshGraphormer 导入成功')
"

# 返回 HandRefiner 主目录
cd ..
pwd  # 应显示 .../HandRefiner
```

---

## 📥 第三步：下载 HandRefiner 模型权重

### 3.1 创建模型文件夹

```bash
# 确保在 HandRefiner 目录
cd ~/HandRefiner

mkdir -p models
ls -la models/
```

### 3.2 下载微调权重（推荐方案）

```bash
cd ~/HandRefiner/models

# 方法 1：使用 aria2 加速下载（推荐）
aria2c -x 5 "https://drive.google.com/uc?id=1eD2Lnfk0KZols68mVahcVfNx3GnYdHxo" -o inpaint_depth_control.ckpt

# 方法 2：使用 wget 下载
wget "https://drive.google.com/uc?id=1eD2Lnfk0KZols68mVahcVfNx3GnYdHxo" -O inpaint_depth_control.ckpt

# 验证下载
ls -lh inpaint_depth_control.ckpt  # 应显示 ~4.5GB
```

### 3.3 验证模型结构

```bash
# 返回 HandRefiner 目录
cd ~/HandRefiner

# 检查模型目录结构
tree models -L 1  # 或 ls -la models/
```

**预期结构**:
```
models/
├── inpaint_depth_control.ckpt  (~4.5GB)  [可选：如果使用方案 A]
├── graphormer_release/  [来自 MeshGraphormer]
│   └── graphormer_hand_state_dict.bin
└── hrnet/  [来自 MeshGraphormer]
    ├── hrnetv2_w64_imagenet_pretrained.pth
    └── cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml
```

---

## 📝 第四步：安装 MediaPipe 和预处理器

```bash
# 确保在 HandRefiner 目录
cd ~/HandRefiner

# 安装 MediaPipe
pip install -q mediapipe==0.10.0

# 下载手部检测模型
cd preprocessor
wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

# 验证
ls -lh hand_landmarker.task

# 返回主目录
cd ..
```

---

## ✅ 第五步：完整环境验证

```bash
# 确保虚拟环境激活
source ~/HandRefiner/venv_gpu/bin/activate

# 进入项目目录
cd ~/HandRefiner

# 运行完整验证脚本
python << 'EOF'
import sys
print("=" * 60)
print("HandRefiner 云端环境验证")
print("=" * 60)

# 检查 PyTorch
import torch
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA 可用: {torch.cuda.is_available()}")
print(f"✅ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无'}")
print()

# 检查主要库
try:
    import cv2
    print(f"✅ OpenCV: {cv2.__version__}")
except: print("❌ OpenCV 缺失")

try:
    import numpy as np
    print(f"✅ NumPy: {np.__version__}")
except: print("❌ NumPy 缺失")

try:
    import mediapipe
    print(f"✅ MediaPipe 已安装")
except: print("❌ MediaPipe 缺失")

try:
    import transformers
    print(f"✅ Transformers: {transformers.__version__}")
except: print("❌ Transformers 缺失")

# 检查模型文件
import os
print()
print("模型文件检查:")
models_path = "models/inpaint_depth_control.ckpt"
if os.path.exists(models_path):
    size_gb = os.path.getsize(models_path) / (1024**3)
    print(f"✅ 微调权重: {size_gb:.2f} GB")
else:
    print("⚠️ 微调权重未找到（使用原始权重时可忽略）")

mg_model = "MeshGraphormer/models/graphormer_release/graphormer_hand_state_dict.bin"
if os.path.exists(mg_model):
    print(f"✅ MeshGraphormer 权重已找到")
else:
    print("❌ MeshGraphormer 权重缺失")

print()
print("=" * 60)
print("验证完成！")
print("=" * 60)
EOF
```

---

## 🧪 第六步：测试单张图像

```bash
# 创建测试输出目录
mkdir -p output

# 获取测试图像（如果没有）
# 或从本地上传一张图像到 test/ 文件夹

# 运行 HandRefiner（单张图像）
python handrefiner.py \
  --input_img test/1.jpg \
  --out_dir output \
  --strength 0.55 \
  --weights models/inpaint_depth_control.ckpt \
  --prompt "a man facing the camera, making a hand gesture, indoor" \
  --seed 1

# 查看结果
ls -lh output/
```

---

## 📊 第七步：批量测试（可选）

```bash
# 准备提示文件 test/test.json（如果尚未存在）
cat > test/test.json << 'EOF'
{"img": "1.jpg", "txt": "a man facing the camera, making a hand gesture"}
{"img": "2.jpg", "txt": "a woman with hands gesturing"}
EOF

# 运行批量处理
python handrefiner.py \
  --input_dir test \
  --out_dir output \
  --strength 0.55 \
  --weights models/inpaint_depth_control.ckpt \
  --prompt_file test/test.json \
  --seed 1

# 查看结果
ls -lh output/
```

---

## 🔄 后续工作流

### 更新本地代码后同步

```bash
# 本地：修改代码并推送
cd c:\Users\86191\Desktop\bs\HandRefiner
git add .
git commit -m "改进：修改参数处理"
git push origin main

# 云端：拉取最新代码
cd ~/HandRefiner
git pull origin main

# 运行更新后的代码
python handrefiner.py --input_img test/1.jpg ...
```

---

## ⚠️ 常见问题

### Q1：MANO 模型无法下载
**解决**:
1. 本地下载后上传
2. 联系 MANO 官方获取下载链接
3. 使用替代手部模型

### Q2：GPU 内存不足
**解决**:
```bash
# 减小批量大小
python handrefiner.py --num_samples 1 ...

# 或降低图像分辨率
```

### Q3：模型下载过慢
**解决**:
```bash
# 使用加速下载工具
pip install aria2
aria2c -x 10 "下载链接"
```

### Q4：导入错误
**解决**:
```bash
# 重新安装依赖
pip install --force-reinstall -r requirements.txt

# 或清理缓存
pip cache purge
```

---

## 📋 快速检查清单

- [ ] ✅ 克隆代码库
- [ ] ✅ 创建虚拟环境并激活
- [ ] ✅ 安装 PyTorch（带 CUDA 支持）
- [ ] ✅ 安装 HandRefiner 依赖
- [ ] ✅ 安装 MeshGraphormer
- [ ] ✅ 复制脚本文件
- [ ] ✅ 下载 MeshGraphormer 模型
- [ ] ✅ 下载 MANO_RIGHT.pkl
- [ ] ✅ 下载 HandRefiner 权重
- [ ] ✅ 安装 MediaPipe
- [ ] ✅ 运行环境验证脚本
- [ ] ✅ 测试单张图像推理
- [ ] ✅ 了解代码同步流程

---

## 🎯 下一步

完成上述所有步骤后，你就可以：

1. ✅ 在本地修改代码并推送到 GitHub
2. ✅ 在云端拉取代码并直接运行
3. ✅ 快速迭代和测试新功能

**愉快的开发！** 🚀
