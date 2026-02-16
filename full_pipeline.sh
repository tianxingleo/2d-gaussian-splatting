#!/bin/bash

# 如果任何命令失败，立即退出
set -e

# ================= 配置 =================
VIDEO_PATH=$1      # 第一个参数：视频文件路径
SCENE_NAME=$2      # 第二个参数：场景名称（如 my_video）
ITERATIONS=${3:-30000} # 第三个参数：迭代次数（可选，默认 30000）

if [ -z "$VIDEO_PATH" ] || [ -z "$SCENE_NAME" ]; then
    echo "用法: ./full_pipeline.sh <视频路径> <场景名称> [迭代次数]"
    echo "示例: ./full_pipeline.sh ~/projects/Sparse2DGS/video.mp4 my_video 7000"
    exit 1
fi

DATA_DIR="data/$SCENE_NAME"
OUTPUT_DIR="output/$SCENE_NAME"

# ================= 环境修复 (参考 scene_pipeline_fixed.py) =================
export QT_QPA_PLATFORM=offscreen
export CUDA_VISIBLE_DEVICES=0
# WSL CUDA 路径修复
if [ -d "/usr/lib/wsl/lib" ]; then
    export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
fi

# 优先使用系统的 colmap (包含 global_mapper 且通常路径已配置好)
COLMAP_EXE="/usr/local/bin/colmap"
if [ ! -f "$COLMAP_EXE" ]; then
    COLMAP_EXE="colmap"
fi

if ! $COLMAP_EXE help | grep -q "global_mapper"; then
    echo "⚠️ 警告: $COLMAP_EXE 不支持 glomap (global_mapper)，将退回到标准增量式重建 (mapper)。"
    USE_INCREMENTAL=true
fi

echo "==== 🚀 [1/5] 准备数据目录和抽帧 ===="
# 清理可能存在的旧数据（可选，为了干净）
rm -rf "$DATA_DIR"

mkdir -p "$DATA_DIR/input"
# 采样率设置为 2fps
# 自动适配横竖屏：横屏缩放到 1920x1080，竖屏缩放到 1080x1920
ffmpeg -i "$VIDEO_PATH" -vf "fps=1,scale='if(gt(iw,ih),1920,1080)':'if(gt(iw,ih),1080,1920)':force_original_aspect_ratio=decrease" "$DATA_DIR/input/%04d.jpg" -y

echo "==== 🔍 [2/5] 运行位姿估计 (SfM) ===="
mkdir -p "$DATA_DIR/distorted/sparse"

# 1. 特征提取
$COLMAP_EXE feature_extractor \
    --database_path "$DATA_DIR/distorted/database.db" \
    --image_path "$DATA_DIR/input" \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model OPENCV \
    --FeatureExtraction.use_gpu 1

# 2. 特征匹配 (对于视频，sequential 往往更快且足够)
$COLMAP_EXE exhaustive_matcher \
    --database_path "$DATA_DIR/distorted/database.db" \
    --FeatureMatching.use_gpu 1

# 3. 重建 (GLOMAP 或标准 Mapper)
if [ "$USE_INCREMENTAL" = true ]; then
    echo "使用增量式重建 (Incremental Mapper)..."
    $COLMAP_EXE mapper \
        --database_path "$DATA_DIR/distorted/database.db" \
        --image_path "$DATA_DIR/input" \
        --output_path "$DATA_DIR/distorted/sparse"
else
    echo "使用 GLOMAP 全局重建 (Global Mapper)..."
    # GLOMAP
    $COLMAP_EXE global_mapper \
        --database_path "$DATA_DIR/distorted/database.db" \
        --image_path "$DATA_DIR/input" \
        --output_path "$DATA_DIR/distorted/sparse"
fi

# 4. 图像去畸变
mkdir -p "$DATA_DIR/sparse"
$COLMAP_EXE image_undistorter \
    --image_path "$DATA_DIR/input" \
    --input_path "$DATA_DIR/distorted/sparse/0" \
    --output_path "$DATA_DIR" \
    --output_type COLMAP

# 5. 整理位姿文件到 0 目录
mkdir -p "$DATA_DIR/sparse/0"
mv "$DATA_DIR/sparse"/*.bin "$DATA_DIR/sparse/0/" 2>/dev/null || true

echo "==== 🛡️ [3/5] 检测 COLMAP/GLOMAP 是否成功 ===="
SPARSE_DIR="$DATA_DIR/sparse/0"

# 检查关键文件是否存在
if [ ! -f "$SPARSE_DIR/cameras.bin" ]; then
    echo "❌ 错误: COLMAP 未能生成位姿文件 (sparse/0/cameras.bin)。"
    exit 1
fi

# 检查点云大小 (如果小于 10KB，说明点太少了，基本是失败的)
POINTS_SIZE=$(stat -c%s "$SPARSE_DIR/points3D.bin")
if [ "$POINTS_SIZE" -lt 10240 ]; then
    echo "❌ 错误: COLMAP 生成的点云太稀疏 ($POINTS_SIZE bytes)，训练将无法收敛。"
    echo "提示: 请尝试增加抽帧频率 (fps) 或拍摄位移更大的视频。"
    exit 1
fi

# 检查注册的图片数量 (通过 images.bin 的大小简单估算，或者直接看输出)
echo "✅ COLMAP 重建成功！(点云大小: $((POINTS_SIZE/1024)) KB)"

echo "==== 🧠 [4/5] 整理完成，准备训练 ===="
# 此时数据结构应为:
# $DATA_DIR/images (去畸变后的图)
# $DATA_DIR/sparse/0 (位姿)

echo "==== 🏋️ [5/5] 开始 2DGS 训练 ===="
# 检查端口 6009 是否被占用，如果占用则杀掉进程防止报错
PORT_PID=$(netstat -nlp 2>/dev/null | grep :6009 | awk '{print $7}' | cut -d'/' -f1)
if [ ! -z "$PORT_PID" ]; then
    echo "检测到端口 6009 被进程 $PORT_PID 占用，正在清理..."
    kill -9 $PORT_PID 2>/dev/null || true
fi

python train.py -s "$DATA_DIR" -m "$OUTPUT_DIR" --iterations "$ITERATIONS"

echo "==== ✨ 全部流程已完成！ ===="
echo "你可以运行以下命令查看效果:"
echo "python view.py -s $DATA_DIR -m $OUTPUT_DIR"
