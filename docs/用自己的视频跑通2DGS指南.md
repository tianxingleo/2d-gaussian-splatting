# 用自己的视频跑通 2DGS 完整指南

## 完整 Pipeline 流程图

```
视频文件 (video.mp4)
    ↓
[1] 视频转图片 (ffmpeg)
    ↓
图片序列 (images/*.jpg)
    ↓
[2] COLMAP 特征提取 (GPU)
    ↓
特征点数据库 (database.db)
    ↓
[3] COLMAP 特征匹配 (GPU)
    ↓
匹配结果 (database.db)
    ↓
[4] COLMAP 稀疏重建 (CPU + GPU)
    ↓
稀疏点云 + 相机位姿 (sparse/0)
    ↓
[5] COLMAP 稠密重建 (可选，GPU)
    ↓
稠密点云 (dense/)
    ↓
[6] 图像下采样 (2DGS 要求)
    ↓
处理后的图片 (images_2x/)
    ↓
[7] 2DGS 训练 (GPU)
    ↓
高斯模型 (output/scene_name/)
    ↓
[8] 2DGS 渲染/可视化 (GPU)
    ↓
渲染结果 + 网格模型
```

---

## 详细步骤说明

### 阶段 1：数据准备 (COLMAP)

#### 1.1 视频转图片

```bash
# 创建工作目录
mkdir -p data/my_video/images

# 提取视频帧（调整 fps 和分辨率）
ffmpeg -i video.mp4 -vf "fps=2,scale=1920:-2" data/my_video/images/%04d.jpg -y
```

**说明：**
- `fps=2`: 每秒提取 2 帧（可调整，通常 2-5 fps 足够）
- `scale=1920:-2`: 保持宽高比，宽度 1920px
- 提取过多帧会导致 COLMAP 处理时间大幅增加

#### 1.2 COLMAP 特征提取

```bash
cd data/my_video

# 提取 SIFT 特征（GPU 加速）
colmap feature_extractor \
    --database_path database.db \
    --image_path images
```

**输出：**
- `database.db`: 包含所有图像的 SIFT 特征点
- 特征点数量：每张图片 1000-8000 个不等

#### 1.3 COLMAP 特征匹配

```bash
# 特征匹配（GPU 加速）
colmap exhaustive_matcher \
    --database_path database.db
```

**说明：**
- 匹配所有图像对（exhaustive）
- 对于 324 张图片，会匹配 52476 对（324×323/2）
- 时间：根据图片数量和分辨率，通常 1-10 分钟

#### 1.4 COLMAP 稀疏重建 (Sparse Reconstruction)

```bash
# 稀疏重建（CPU 密集计算 + GPU 加速）
colmap mapper \
    --database_path database.db \
    --image_path images \
    --output_path sparse
```

**说明：**
- **这是最耗时的步骤**，主要在 CPU 上运行
- Bundle Adjustment 优化相机位姿和 3D 点云
- 时间：30 分钟到数小时不等
- 输出：`sparse/0/` 目录包含相机参数、点云、图像

**CPU vs GPU 使用：**
- ✅ 特征提取：GPU (SIFT GPU matcher)
- ✅ 特征匹配：GPU (SIFT GPU matcher)
- ⚠️ 稀疏重建：**主要是 CPU** (Bundle Adjustment 是 CPU 密集计算)
- ✅ 稠密重建：GPU (Stereo matching, Fusion)

#### 1.5 稠密重建 (可选，Dense Reconstruction)

```bash
# 稠密重建（GPU）
colmap image_undistorter \
    --image_path images \
    --input_path sparse/0 \
    --output_path dense

colmap stereo_fusion \
    --workspace_path dense \
    --workspace_format COLMAP

colmap poisson_mesher \
    --input_path dense/fused.ply \
    --output_path dense/meshed-poisson.ply
```

**说明：**
- **可选步骤**，2DGS 不需要稠密重建
- 主要用于生成网格模型或可视化
- 时间：较长，需要大量显存

---

### 阶段 2：2DGS 训练准备

#### 2.1 图像下采样（必需！）

**重要：2DGS 需要原始分辨率 2x 下采样的图片**

```bash
# 方法 1: 使用 COLMAP 图像调节器
colmap image_undistorter \
    --image_path images \
    --input_path sparse/0 \
    --output_path images_2x \
    --max_image_size 2000  # 根据原始分辨率调整

# 方法 2: 使用 ffmpeg 下采样
mkdir -p images_2x
ffmpeg -i images/%04d.jpg -vf "scale=iw/2:ih/2" images_2x/%04d.jpg
```

**说明：**
- 2DGS 论文要求训练使用 2x 下采样的图像
- 测试时使用原始分辨率

#### 2.2 目录结构要求

```
data/my_video/
├── images/           # 原始分辨率图片（用于测试）
├── images_2x/        # 2x 下采样图片（用于训练）
├── sparse/
│   └── 0/
│       ├── cameras.bin      # 相机参数
│       ├── images.bin       # 图像信息
│       └── points3D.bin    # 稀疏点云
└── database.db        # COLMAP 数据库
```

---

### 阶段 3：2DGS 训练

#### 3.1 基础训练

```bash
cd ~/projects/2d-gaussian-splatting

# 激活环境
conda activate gs_linux_backup

# 训练
python train.py -s data/my_video -m output/my_video
```

**参数说明：**
- `-s data/my_video`: 数据集路径（包含 images 和 sparse 目录）
- `-m output/my_video`: 输出模型路径
- 默认使用 2x 下采样图片训练

**训练时间：**
- 取决于图片数量、分辨率、GPU 性能
- 小场景（100-200 张）：10-30 分钟
- 大场景（300-500 张）：30 分钟 - 2 小时

#### 3.2 高级训练参数

```bash
# 使用正则化项
python train.py \
    -s data/my_video \
    -m output/my_video \
    --lambda_normal 0.05 \      # 法向一致性正则化
    --lambda_distortion 0.01 \   # 深度畸变正则化
    --depth_ratio 0              # 0: 均值深度, 1: 中位数深度
```

**正则化说明：**
- `lambda_normal`: 法向一致性，提高几何质量（推荐 0.01-0.1）
- `lambda_distortion`: 深度畸变，减少伪影（推荐 0.005-0.05）
- `depth_ratio`: 深度计算方式
  - `0`: 均值深度（适合有界场景）
  - `1`: 中位数深度（适合无界场景，如 MipNeRF360）

---

### 阶段 4：2DGS 渲染与可视化

#### 4.1 渲染新视角

```bash
# 渲染训练集
python render.py -m output/my_video -s data/my_video

# 渲染测试集（如果有）
python render.py -m output/my_video -s data/my_video --skip_train
```

#### 4.2 提取网格模型

**有界网格提取 (Bounded Mesh):**

```bash
python render.py \
    -m output/my_video \
    -s data/my_video \
    --mesh_res 512 \              # 体素分辨率（更高 = 更精细）
    --voxel_size 0.02 \           # 体素大小
    --depth_trunc 0.1 \           # 深度截断
    --skip_test --skip_train        # 只提取网格，不渲染
```

**无界网格提取 (Unbounded Mesh):**

```bash
python render.py \
    -m output/my_video \
    -s data/my_video \
    --unbounded \                  # 无界模式
    --mesh_res 1024 \             # 更高分辨率
    --skip_test --skip_train
```

**说明：**
- 有界模式：适合室内场景、前景物体
- 无界模式：适合室外大场景、MipNeRF360 风格
- 输出：`output/my_video/mesh.ply`

#### 4.3 实时可视化

**使用 SIBR Viewer:**

```bash
# 1. 启动 SIBR Viewer
<path_to_viewer>/bin/SIBR_remoteGaussian_app_rwdi

# 2. 监控训练过程（训练时自动连接）
python train.py -s data/my_video -m output/my_video

# 3. 查看训练好的模型
python view.py -s data/my_video -m output/my_video
```

---

## 常见问题

### Q1: COLMAP 稀疏重建太慢怎么办？

**A:** 稀疏重建主要在 CPU 上运行，无法加速。可以：

1. **减少图片数量**：
   ```bash
   # 提取更少的帧（fps=1）
   ffmpeg -i video.mp4 -vf "fps=1,scale=1920:-2" images/%04d.jpg
   ```

2. **降低分辨率**：
   ```bash
   # 使用更低的分辨率
   ffmpeg -i video.mp4 -vf "fps=2,scale=1280:-2" images/%04d.jpg
   ```

3. **使用顺序匹配代替穷举匹配**（更快但可能遗漏一些图像）：
   ```bash
   colmap sequential_matcher --database_path database.db
   ```

### Q2: 训练不收敛怎么办？

**A:** 检查以下几点：

1. **COLMAP 质量差**：检查 `sparse/0/` 中的点云是否足够密集
2. **相机参数错误**：确保主点在图像中心附近
3. **学习率太高**：尝试降低 `--densify_until_iter` 参数
4. **使用正则化**：添加 `--lambda_normal` 和 `--lambda_distortion`

### Q3: 如何处理自己的视频？

**A:** 关键点：

1. **相机移动**：视频必须有相机运动（平移、旋转），不能是纯旋转
2. **光照稳定**：避免剧烈光照变化
3. **场景清晰**：避免运动模糊、低光照
4. **帧数控制**：
   - 小物体/室内：200-300 帧
   - 大场景/室外：300-500 帧
5. **采样策略**：
   - 均匀采样整个视频（不要集中在前几秒）
   - `ffmpeg -ss 0 -to 60 -i video.mp4` 提取前 1 分钟

### Q4: 5070 显卡需要注意什么？

**A:** RTX 5070 是较新的架构，需要注意：

1. **CUDA 版本**：确保 COLMAP 和 PyTorch 支持 CUDA 12.8+
2. **显存**：12GB 足够中小场景，大场景可能需要降低分辨率
3. **编译问题**：某些 CUDA 扩展可能需要针对新 GPU 编译

---

## 当前任务进度

### 已完成 ✅

1. ✅ 视频转图片：324 帧（2 fps）
   - 路径：`data/my_video/images/*.jpg`
   - 分辨率：1920x3414

2. ✅ COLMAP 特征提取：324 张图片
   - 输出：`database.db`
   - 每张图片特征：1000-8128 个

3. ✅ COLMAP 特征匹配：所有图像对
   - 匹配对数：52476 对（324×323/2）
   - 时间：约 1.1 分钟

### 进行中 ⏳

4. ⏳ COLMAP 稀疏重建 (Running)
   - 已注册：150/324 张图像（46%）
   - 当前图像点数：112-8128
   - **主要在 CPU 上运行**（Bundle Adjustment）
   - 预计剩余时间：10-30 分钟

### 待完成 ⏸

5. ⏸ 图像下采样到 images_2x/
6. ⏸ 2DGS 训练
7. ⏸ 渲染和可视化

---

## CPU vs GPU 使用情况总结

| 步骤 | CPU 使用 | GPU 使用 | 说明 |
|------|---------|---------|------|
| 视频转图片 | 中 | 低 | ffmpeg 主要用 CPU |
| COLMAP 特征提取 | 中 | **高** | SIFT GPU 加速 |
| COLMAP 特征匹配 | 低 | **高** | SIFT GPU 加速 |
| COLMAP 稀疏重建 | **极高** | 低 | Bundle Adjustment 是 CPU 密集计算 |
| COLMAP 稠密重建 | 中 | **高** | Stereo matching, Fusion |
| 2DGS 训练 | 低 | **极高** | 完全 GPU 加速 |
| 2DGS 渲染 | 低 | **高** | 完全 GPU 加速 |

---

## 快速开始命令（参考）

```bash
# 1. 激活环境
conda activate gs_linux_backup

# 2. 进入项目目录
cd ~/projects/2d-gaussian-splatting

# 3. 训练（COLMAP 完成后）
python train.py -s data/my_video -m output/my_video

# 4. 渲染
python render.py -m output/my_video -s data/my_video

# 5. 提取网格
python render.py -m output/my_video -s data/my_video \
    --mesh_res 512 --skip_test --skip_train
```

---

**文档更新时间：** 2026-02-16
**环境：** WSL2 + Ubuntu + RTX 5070 + CUDA 12.8
