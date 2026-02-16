# -*- coding: utf-8 -*-
"""
场景重建 Pipeline（修复版）：从手机视频到 Sparse2DGS
结合 BrainDance 的抽帧+COLMAP重建流程与 Sparse2DGS 训练

修复内容：
1. 使用系统的 colmap（强制使用 /usr/local/bin/colmap）
2. 添加更好的错误处理和日志
3. 修复 global_mapper 命令调用问题
"""
import subprocess
import sys
import shutil
import os
import time
import datetime
from pathlib import Path
import json
import numpy as np
import cv2
import re
import struct

# 简单的 COLMAP 二进制读取工具，避免复杂的导入
def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file."""
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

# ================= 配置 =================
LINUX_WORK_ROOT = Path.home() / "scene_reconstruction"
MAX_IMAGES = 50  # 场景重建需要更多视角
FPS = 4  # 抽帧率
VIDEO_SCALE = 1920  # 视频缩放
KEEP_PERCENTILE = 0.5  # 采样率 (预留字段)

# Sparse2DGS 相关配置
SPARSE2DGS_PATH = Path("/home/ltx/projects/Sparse2DGS")

# ================= 辅助工具：时间格式化 =================
def format_duration(seconds):
    """将秒数转换为 HH:MM:SS 格式"""
    return str(datetime.timedelta(seconds=int(seconds)))

# ================= 辅助工具：模糊图片过滤 =================
def smart_filter_blurry_images(image_folder, keep_ratio=0.90, max_images=MAX_IMAGES):
    """
    升级版清洗脚本：混合策略 (Hybrid Strategy)
    """
    print(f"\n🧠 [智能清洗] 正在分析图片质量 (混合策略版)...")
    
    image_dir = Path(image_folder)
    images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    
    if not images:
        print("❌ 没找到图片")
        return

    trash_dir = image_dir.parent / "trash_smart"
    trash_dir.mkdir(exist_ok=True)

    img_scores = []

    # 第一步：计算分数
    for i, img_path in enumerate(images):
        img = cv2.imread(str(img_path))
        if img is None: continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # 九宫格评分
        grid_h, grid_w = h // 3, w // 3
        max_grid_score = 0
        for r in range(3):
            for c in range(3):
                roi = gray[r*grid_h:(r+1)*grid_h, c*grid_w:(c+1)*grid_w]
                score = cv2.Laplacian(roi, cv2.CV_64F).var()
                if score > max_grid_score:
                    max_grid_score = score
        
        img_scores.append((img_path, max_grid_score))
        if i % 20 == 0:
            print(f"  -> 分析中... {img_path.name}: 局部最高分 {max_grid_score:.1f}")

    # 第二步：质量清洗
    scores = [s[1] for s in img_scores]
    if not scores: return

    num_total = len(scores)
    quality_threshold = np.percentile(scores, (1 - keep_ratio) * 100)
    
    print(f"\n📊 统计结果:")
    print(f"   - 图片总数: {num_total}")
    print(f"   - 质量阈值 (Bottom {(1-keep_ratio)*100:.0f}%): {quality_threshold:.2f}")

    good_images = []
    removed_count_quality = 0

    for img_path, score in img_scores:
        if score < quality_threshold:
            shutil.move(str(img_path), str(trash_dir / img_path.name))
            removed_count_quality += 1
        else:
            good_images.append(img_path)

    print(f"   -> 第一轮清洗完成: 剔除 {removed_count_quality} 张废片，剩余 {len(good_images)} 张。")

    # 第三步：数量控制（均匀采样）
    removed_count_quantity = 0
    
    if len(good_images) > max_images:
        print(f"   ⚠️ 合格图片 ({len(good_images)}) 仍超过上限 ({max_images})")
        print(f"   -> 执行【均匀采样】以保证视角覆盖...")
        
        indices_to_keep = set(np.linspace(0, len(good_images) - 1, max_images, dtype=int))
        
        for idx, img_path in enumerate(good_images):
            if idx not in indices_to_keep:
                shutil.move(str(img_path), str(trash_dir / img_path.name))
                removed_count_quantity += 1
    else:
        print(f"   ✅ 合格图片数量 ({len(good_images)}) 未超标，全部保留。")

    total_removed = removed_count_quality + removed_count_quantity
    final_count = num_total - total_removed
    print(f"✨ 清洗结束: 共移除 {total_removed} 张 (废片 {removed_count_quality} + 采样 {removed_count_quantity})，最终保留 {final_count} 张。")

# ================= 辅助工具：运行命令 =================
def run_command(cmd, description, env=None, cwd=None):
    """运行命令并输出日志"""
    print(f"\n⚡ {description}...")
    try:
        with subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True, 
            env=env or os.environ.copy(),
            bufsize=1,
            cwd=cwd
        ) as process:
            for line in process.stdout:
                print(line, end='')
                # 实时刷新输出
                sys.stdout.flush()
            
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)
    except Exception as e:
        print(f"❌ {description} 失败: {e}")
        raise e

# ================= 辅助工具：生成 DTU 格式相机文件 (为 CLMVSNet 准备) =================
def generate_dtu_cameras(colmap_sparse_dir, output_dtu_dir):
    """
    从 COLMAP sparse 目录读取数据，并生成 DTU 格式的 cam_*.txt 文件
    CLMVSNet 训练需要这些文件中的 dp_min 和 dp_max
    """
    print(f"    -> 正在生成 DTU 格式相机参数 (供 MVS 深度估计使用)...")
    
    colmap_sparse_dir = Path(colmap_sparse_dir)
    output_dtu_dir = Path(output_dtu_dir)
    output_dtu_dir.mkdir(parents=True, exist_ok=True)
    
    # 路径
    cam_bin = colmap_sparse_dir / "cameras.bin"
    img_bin = colmap_sparse_dir / "images.bin"
    pts_bin = colmap_sparse_dir / "points3D.bin"
    
    if not (cam_bin.exists() and img_bin.exists() and pts_bin.exists()):
        print(f"    ⚠️ 缺少 COLMAP 二进制文件，跳过 DTU 相机生成")
        return False

    # 1. 读取相机内参
    intrinsics = {}
    with open(str(cam_bin), "rb") as f:
        num_cameras = read_next_bytes(f, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_id, model_id, width, height = read_next_bytes(f, 24, "iiQQ")
            num_params = 0
            if model_id == 0: num_params = 3 # SIMPLE_PINHOLE
            elif model_id == 1: num_params = 4 # PINHOLE
            elif model_id == 2: num_params = 2 # SIMPLE_RADIAL
            elif model_id == 4: num_params = 8 # OPENCV
            
            params = read_next_bytes(f, 8 * num_params, "d" * num_params)
            
            # 简化：只提取 K 矩阵
            K = np.eye(3)
            if model_id == 0: # SIMPLE_PINHOLE: f, cx, cy
                K[0,0] = K[1,1] = params[0]
                K[0,2], K[1,2] = params[1], params[2]
            elif model_id == 1: # PINHOLE: fx, fy, cx, cy
                K[0,0], K[1,1] = params[0], params[1]
                K[0,2], K[1,2] = params[2], params[3]
            else: # 兜底
                K[0,0] = K[1,1] = params[0]
                K[0,2], K[1,2] = width/2, height/2
                
            intrinsics[camera_id] = K

    # 2. 读取 3D 点云 (为了计算深度范围)
    with open(str(pts_bin), "rb") as f:
        num_points = read_next_bytes(f, 8, "Q")[0]
        xyzs = np.empty((num_points, 3))
        for i in range(num_points):
            binary_point_line_properties = read_next_bytes(f, 43, "QdddBBBd")
            xyzs[i] = binary_point_line_properties[1:4]
            track_length = read_next_bytes(f, 8, "Q")[0]
            f.seek(8 * track_length, 1) # 跳过 track

    # 3. 读取图像外参并生成文件
    with open(str(img_bin), "rb") as f:
        num_images = read_next_bytes(f, 8, "Q")[0]
        for _ in range(num_images):
            image_id, qw, qx, qy, qz, tx, ty, tz, camera_id = read_next_bytes(f, 64, "idddddddi")
            image_name = ""
            while True:
                char = f.read(1).decode("utf-8")
                if char == "\0": break
                image_name += char
            
            num_points2d = read_next_bytes(f, 8, "Q")[0]
            xys_point3d_ids = read_next_bytes(f, 24 * num_points2d, "ddq" * num_points2d)
            
            # 获取有效的 3D 点 id
            point3d_ids = []
            for i in range(num_points2d):
                p_id = xys_point3d_ids[i*3 + 2]
                if p_id != -1: point3d_ids.append(p_id)
            
            # 计算外参矩阵 w2c
            R = qvec2rotmat([qw, qx, qy, qz])
            T = np.array([tx, ty, tz])
            
            w2c = np.eye(4)
            w2c[:3, :3] = R
            w2c[:3, 3] = T
            
            # 计算深度范围 (dp_min, dp_max)
            # Heuristic: 如果有可见点，基于可见点计算；否则使用全局点云的统计值
            if len(point3d_ids) > 10:
                # 这种方法比较慢，我们简单采样一些点
                sample_ids = np.random.choice(point3d_ids, min(500, len(point3d_ids)), replace=False)
                # 由于 xyzs 的索引不是 point3d_id，我们需要特殊处理。
                # 但在 COLMAP 二进制中，xyzs 的顺序不一定对应 id。
                # 为简单起见，我们使用一个粗略的范围：基于所有点的投影
                pass 
            
            # 粗略方案：使用所有 3D 点投影到相机的深度
            # 为了性能，只对前 1000 个点计算
            pts_sample = xyzs[::max(1, len(xyzs)//1000)]
            pts_cam = (R @ pts_sample.T).T + T
            depths = pts_cam[:, 2]
            depths = depths[depths > 0] # 只要相机前方的点
            
            if len(depths) > 0:
                dp_min = np.percentile(depths, 5) * 0.8
                dp_max = np.percentile(depths, 95) * 1.2
            else:
                dp_min, dp_max = 0.1, 10.0 # 兜底值
            
            # 获取重建的分辨率 (实际存储的分辨率)
            try:
                from PIL import Image
                with Image.open(str(images_dir / image_name)) as img:
                    actual_width, actual_height = img.size
            except:
                actual_width, actual_height = width, height

            # 写入文件
            K = intrinsics[camera_id].copy()
            # 如果实际分辨率和 COLMAP 记录的分辨率不一致 (因为我们压缩了图像)，则需要缩放内参
            if width != actual_width or height != actual_height:
                scale_x = actual_width / width
                scale_y = actual_height / height
                K[0,0] *= scale_x
                K[1,1] *= scale_y
                K[0,2] *= scale_x
                K[1,2] *= scale_y

            txt_name = f"cam_{Path(image_name).stem}.txt"
            with open(str(output_dtu_dir / txt_name), "w") as tf:
                # K
                for row in K: tf.write(f"{row[0]} {row[1]} {row[2]}\n")
                tf.write("\n")
                # w2c
                for row in w2c: tf.write(f"{row[0]} {row[1]} {row[2]} {row[3]}\n")
                tf.write("\n")
                # depth range
                tf.write(f"{dp_min} {dp_max}\n")
    
    print(f"    ✅ 已生成 {num_images} 个 DTU 格式相机文件")
    return True

# ================= 辅助工具：准备 Sparse2DGS 数据 =================
def prepare_sparse2dgs_data(colmap_output, target_dir, scene_name):
    """
    准备 Sparse2DGS 数据格式
    """
    print(f"\n📦 [数据转换] 准备 Sparse2DGS 数据...")
    
    target_dir = Path(target_dir)
    scene_dir = target_dir / scene_name
    images_dir = scene_dir / "images"
    # Sparse2DGS 兼容性：创建 sparse/0 结构
    sparse_target_dir = scene_dir / "sparse" / "0"
    
    # 创建目录结构
    scene_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_target_dir.mkdir(parents=True, exist_ok=True)
    
    # 复制图像并进行 4K -> 2K 压缩以节省内存 (建议)
    colmap_images = colmap_output / "raw_images"
    if not colmap_images.exists():
        colmap_images = colmap_output / "images"
    
    image_count = 0
    if colmap_images.exists():
        print(f"    📷 正在处理图像 (保持高质量但限制最大边长为 2048 以节省内存)...")
        from PIL import Image
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]:
            for img_path in colmap_images.glob(ext):
                target_path = images_dir / img_path.name
                try:
                    with Image.open(img_path) as img:
                        # 如果图像过大，将其等比例缩小到 720P 水准 (1280px)
                        # 这将极大降低 RAM / VRAM 开发，防止系统崩溃
                        max_dim = 1280
                        if img.width > max_dim or img.height > max_dim:
                            scale = max_dim / max(img.width, img.height)
                            new_size = (int(img.width * scale), int(img.height * scale))
                            img = img.resize(new_size, Image.Resampling.LANCZOS)
                        img.save(target_path, quality=95)
                    image_count += 1
                except Exception as e:
                    print(f"    ⚠️ 图像 {img_path.name} 处理失败: {e}")
                    shutil.copy2(str(img_path), str(target_path))
                    image_count += 1
                image_count += 1
    
    print(f"    ✅ 已复制 {image_count} 张图像")
    
    # 查找并复制 COLMAP sparse 数据
    possible_dirs = [
        colmap_output / "sparse" / "0",
        colmap_output / "sparse",
        colmap_output / "colmap_output" / "sparse" / "0",
        colmap_output / "colmap_output" / "sparse",
    ]
    
    src_sparse_dir = None
    for d in possible_dirs:
        if d.exists() and (d / "cameras.bin").exists():
            src_sparse_dir = d
            break
            
    # 兜底搜索
    if not src_sparse_dir:
        for root, dirs, files in os.walk(colmap_output):
            if "cameras.bin" in files and "images.bin" in files:
                src_sparse_dir = Path(root)
                break
    
    if src_sparse_dir:
        copy_count = 0
        for file in src_sparse_dir.glob("*"):
            if file.suffix in ['.bin', '.txt', '.ini']:
                shutil.copy2(str(file), str(sparse_target_dir / file.name))
                copy_count += 1
        print(f"    ✅ 已从 {src_sparse_dir.name} 复制 {copy_count} 个数据文件到 sparse/0")
        
        # --- 新增：为 Sparse2DGS 的 MVS 模块生成 DTU 格式相机文件 ---
        dtu_dir = SPARSE2DGS_PATH / "dtu_sparse" / scene_name
        generate_dtu_cameras(src_sparse_dir, dtu_dir)
        
        return scene_dir
    else:
        print("❌ 未找到任何有效的 COLMAP sparse 数据 (cameras.bin/images.bin)")
        return None

# ================= 辅助工具：运行 Sparse2DGS 训练 =================
def run_sparse2dgs_training(scene_dir, output_dir, scan_name, env=None):
    """
    运行 Sparse2DGS 训练
    """
    print(f"\n🚀 [Sparse2DGS] 开始训练...")
    
    # 训练脚本路径
    train_script = SPARSE2DGS_PATH / "train.py"
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 准备参数
    args = [
        "python", str(train_script),
        "--source_path", str(scene_dir),
        "--model_path", str(output_dir / scan_name),
        "--images", "images",
        "--eval",
        "--iterations", "30000",
        "--resolution", "1",
        "--white_background",
    ]
    
    # 运行训练
    run_command(args, "训练 Sparse2DGS", cwd=str(SPARSE2DGS_PATH), env=env)
    
    print(f"\n✅ Sparse2DGS 训练完成！")
    print(f"   输出目录: {output_dir / scan_name}")
    
    return output_dir / scan_name

# ================= 主流程 =================

def run_pipeline(video_path, project_name):
    """完整的场景重建 Pipeline"""
    
    global_start_time = time.time()
    print(f"\n🚀 [场景重建 Pipeline v2.0] 启动任务: {project_name}")
    print(f"🕒 开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔪 切割策略: 保留 {KEEP_PERCENTILE*100}% 最近点云")
    
    video_src = Path(video_path).resolve()
    work_dir = LINUX_WORK_ROOT / project_name
    colmap_output_dir = work_dir / "colmap_output"
    sparse2dgs_data_dir = work_dir / "sparse2dgs_data"
    sparse2dgs_output_dir = work_dir / "sparse2dgs_output"
    
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"
    
    # WSL CUDA 修复：确保能找到 GPU 驱动
    wsl_lib_path = "/usr/lib/wsl/lib"
    if os.path.exists(wsl_lib_path):
        current_ld_path = env.get("LD_LIBRARY_PATH", "")
        if wsl_lib_path not in current_ld_path:
            env["LD_LIBRARY_PATH"] = f"{wsl_lib_path}:{current_ld_path}".strip(":")
    
    # 显式指定 GPU 设备
    env["CUDA_VISIBLE_DEVICES"] = "0"
    
    print(f"🔧 环境配置: LD_LIBRARY_PATH={env.get('LD_LIBRARY_PATH', 'Not Set')}")
    step1_start = time.time()
    
    print(f"\n🎥 [Step 1/4] 数据准备")
    
    # 创建工作目录
    if work_dir.exists():
        try:
            shutil.rmtree(work_dir)
        except Exception as e:
            print(f"⚠️ 警告: 旧目录清理失败 (可能被占用): {e}")
    
    work_dir.mkdir(parents=True, exist_ok=True)
    colmap_output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(video_src), str(work_dir / video_src.name))

    # 创建临时目录
    temp_dir = work_dir / "temp_extract"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    extracted_images_dir = colmap_output_dir / "raw_images"
    extracted_images_dir.mkdir(parents=True, exist_ok=True)
    
    # FFmpeg 抽帧
    print(f"    -> 正在抽帧...")
    cap = cv2.VideoCapture(str(work_dir / video_src.name))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    
    vf_param = f"fps={FPS}"
    if width > VIDEO_SCALE:
        vf_param = f"scale={VIDEO_SCALE}:-1,fps={FPS}"
    
    try:
        run_command([
            "ffmpeg", "-y", "-i", str(work_dir / video_src.name),
            "-vf", vf_param, "-q:v", "2",
            str(temp_dir / "frame_%05d.jpg")
        ], "抽帧", env=env)
    except Exception as e:
        print(f"⚠️ FFmpeg 抽帧结束: {e}")
    
    # 智能过滤
    smart_filter_blurry_images(temp_dir, keep_ratio=0.90, max_images=MAX_IMAGES)
    
    # 迁移图片
    print(f"    -> 正在迁移图片...")
    all_candidates = sorted(list(temp_dir.glob("*.jpg")) + list(temp_dir.glob("*.png")))
    total_candidates = len(all_candidates)
    
    final_images_list = []
    if total_candidates > MAX_IMAGES:
        print(f"    ⚠️ 图片过多 ({total_candidates}), 正在均匀选取 {MAX_IMAGES} 张...")
        indices = np.linspace(0, total_candidates - 1, MAX_IMAGES, dtype=int)
        indices = sorted(list(set(indices)))
        
        for idx in indices:
            final_images_list.append(all_candidates[idx])
    else:
        final_images_list = all_candidates
    
    for img_path in final_images_list:
        shutil.copy2(str(img_path), str(extracted_images_dir / img_path.name))
    
    print(f"    ✅ 已迁移 {len(final_images_list)} 张图片")
    shutil.rmtree(temp_dir)
    
    step1_duration = time.time() - step1_start
    print(f"⏱️ [Step 1 完成] 耗时: {format_duration(step1_duration)}")
    
    # ================= Step 2: COLMAP 重建 =================
    step2_start = time.time()
    
    print(f"\n🗺️  [Step 2/4] COLMAP Global Mapper 重建 (GLOMAP 功能)")
    
    # 强制使用系统的 colmap（包含 global_mapper）
    system_colmap_exe = "/usr/local/bin/colmap"
    
    # 验证 colmap 是否存在
    if not os.path.exists(system_colmap_exe):
        raise FileNotFoundError(f"❌ 无法找到 colmap: {system_colmap_exe}")
    
    print(f"🎯 COLMAP (包含 global_mapper): {system_colmap_exe}")
    
    database_path = colmap_output_dir / "database.db"
    
    # 特征提取（使用系统 colmap）
    extractor_args = [
        system_colmap_exe, "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(extracted_images_dir),
        "--ImageReader.camera_model", "PINHOLE",
        "--ImageReader.single_camera", "1",
        "--FeatureExtraction.use_gpu", "1"
    ]
    
    try:
        run_command(extractor_args, "特征提取 (COLMAP-GPU)", env=env)
    except subprocess.CalledProcessError as e:
        print(f"\n⚠️ CUDA 初始化失败或参数不兼容，尝试切换到 CPU 模式进行特征提取...")
        extractor_args[-1] = "0" # 将 use_gpu 1 变为 0
        run_command(extractor_args, "特征提取 (COLMAP-CPU)", env=env)
    
    # 顺序匹配（使用系统 colmap）
    matcher_args = [
        system_colmap_exe, "sequential_matcher",
        "--database_path", str(database_path),
        "--SequentialMatching.overlap", "25",
        "--FeatureMatching.use_gpu", "1"
    ]
    
    try:
        run_command(matcher_args, "顺序匹配 (COLMAP-GPU)", env=env)
    except subprocess.CalledProcessError as e:
        print(f"\n⚠️ CUDA 初始化失败或参数不兼容，尝试切换到 CPU 模式进行特征匹配...")
        matcher_args[-1] = "0" # 将 use_gpu 1 变为 0
        run_command(matcher_args, "顺序匹配 (COLMAP-CPU)", env=env)
    
    # COLMAP Global Mapper（GLOMAP 全局重建）
    global_mapper_output_dir = colmap_output_dir / "sparse"
    global_mapper_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"    -> 运行 global_mapper...")
    
    global_mapper_args = [
        system_colmap_exe, "global_mapper",
        "--database_path", str(database_path),
        "--image_path", str(extracted_images_dir),
        "--output_path", str(global_mapper_output_dir)
    ]
    
    try:
        run_command(global_mapper_args, "COLMAP Global Mapper (GLOMAP 全局重建)", env=env)
    except subprocess.CalledProcessError as e:
        print(f"\n⚠️ Global Mapper GPU 模式失败，尝试切换到 CPU 模式...")
        cpu_args = global_mapper_args + [
            "--GlobalMapper.gp_use_gpu", "0",
            "--GlobalMapper.ba_ceres_use_gpu", "0"
        ]
        try:
            run_command(cpu_args, "COLMAP Global Mapper (CPU 模式)", env=env)
        except subprocess.CalledProcessError as e2:
            print(f"❌ COLMAP Global Mapper 完全失败: {e2}")
            
            # 检查输出目录
            if global_mapper_output_dir.exists():
                files = list(global_mapper_output_dir.glob("*.bin")) + list(global_mapper_output_dir.glob("*.txt"))
                if files:
                    print(f"   -> 找到 {len(files)} 个输出文件")
                else:
                    print(f"   -> 输出目录为空")
            
            # 尝试修复逻辑：即使报错，也检查是否生成了模型文件
            required_files = ["cameras.bin", "images.bin", "points3D.bin"]
            sparse_root = colmap_output_dir / "sparse"
            found_repair = False
            for root, dirs, files in os.walk(sparse_root):
                if all(f in files for f in required_files):
                    src_path = Path(root)
                    if not (colmap_output_dir / "sparse" / "0").exists():
                        (colmap_output_dir / "sparse" / "0").mkdir(parents=True, exist_ok=True)
                    for f in required_files:
                        shutil.move(str(src_path / f), str(colmap_output_dir / "sparse" / "0" / f))
                    print(f"   -> 已找到并修复输出文件: {len(required_files)} 个")
                    found_repair = True
                    break
            
            if not found_repair:
                raise e2
    
    # 整理目录结构（COLMAP Global Mapper 输出）
    colmap_sparse_root = colmap_output_dir / "sparse"
    required_files = ["cameras.bin", "images.bin", "points3D.bin"]
    
    # 检查输出位置
    sparse_output = None
    possible_locations = [
        colmap_output_dir / "sparse" / "0",  # 标准位置
        colmap_output_dir / "sparse",         # 根目录
    ]
    
    for loc in possible_locations:
        if all((loc / f).exists() for f in required_files):
            sparse_output = loc
            break
    
    if sparse_output is None:
        # 查找模型文件
        for root, dirs, files in os.walk(colmap_sparse_root):
            if all(f in files for f in required_files):
                src_path = Path(root)
                if not (colmap_output_dir / "sparse" / "0").exists():
                    (colmap_output_dir / "sparse" / "0").mkdir(parents=True, exist_ok=True)
                for f in required_files:
                    shutil.move(str(src_path / f), str(colmap_output_dir / "sparse" / "0" / f))
                sparse_output = colmap_output_dir / "sparse" / "0"
                break
    
    if sparse_output is None:
        raise RuntimeError("COLMAP Global Mapper 重建失败，未找到输出文件")
    
    print(f"✅ COLMAP Global Mapper 重建完成（GLOMAP 功能）")
    
    step2_duration = time.time() - step2_start
    print(f"⏱️ [Step 2 完成] 耗时: {format_duration(step2_duration)}")
    print(f"   输出: {sparse_output}")
    
    # ================= Step 3: 准备 Sparse2DGS 数据 =================
    step3_start = time.time()
    
    print(f"\n📦 [Step 3/4] 准备 Sparse2DGS 数据")
    
    scene_dir = prepare_sparse2dgs_data(colmap_output_dir, sparse2dgs_data_dir, project_name)
    
    if scene_dir is None:
        raise RuntimeError("数据准备失败")
    
    step3_duration = time.time() - step3_start
    print(f"⏱️ [Step 3 完成] 耗时: {format_duration(step3_duration)}")
    
    # ================= Step 4: Sparse2DGS 训练 =================
    step4_start = time.time()
    
    print(f"\n🚀 [Step 4/4] Sparse2DGS 训练")
    
    output_dir = run_sparse2dgs_training(
        scene_dir,
        sparse2dgs_output_dir,
        project_name,
        env=env
    )
    
    step4_duration = time.time() - step4_start
    print(f"⏱️ [Step 4 完成] 耗时: {format_duration(step4_duration)}")
    
    # ================= 完成 =================
    total_time = time.time() - global_start_time
    
    print(f"\n✅ =============================================")
    print(f"🎉 场景重建完成！")
    print(f"📂 最终输出: {output_dir}")
    print(f"⏱️ 总耗时: {format_duration(total_time)}")
    print(f"✅ =============================================")
    
    # 回传结果
    if output_dir and output_dir.exists():
        target_dir = Path(__file__).parent / "results"
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # 复制所有 ply 文件
        for file in output_dir.glob("**/*.ply"):
            shutil.copy2(str(file), str(target_dir / file.name))
            print(f"📦 已复制: {file.name}")
        
        print(f"\n📂 结果已保存到: {target_dir}")
    
    return str(output_dir)

if __name__ == "__main__":
    # 设置编码
    if sys.platform == 'win32':
        import codecs
        sys.stdout = codecs.getwriter(sys.stdout, 'utf-8')('replace')
        sys.stderr = codecs.getwriter(sys.stderr, 'utf-8')('replace')
    
    # 命令行参数
    if len(sys.argv) < 2:
        print("用法: python scene_pipeline_fixed.py <视频路径> [项目名称]")
        print("示例: python scene_pipeline_fixed.py video.mp4 my_scene")
        sys.exit(1)
    
    video_file = Path(sys.argv[1])
    project_name = sys.argv[2] if len(sys.argv) > 2 else "scene_auto"
    
    if not video_file.exists():
        print(f"❌ 找不到视频: {video_file}")
        sys.exit(1)
    
    try:
        run_pipeline(video_file, project_name)
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断 (Ctrl+C)")
        print("正在清理...")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline 失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
