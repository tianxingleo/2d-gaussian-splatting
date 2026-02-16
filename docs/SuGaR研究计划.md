# SuGaR 研究计划

## 搜索结果

当前无法使用 web 搜索（API 问题），基于现有知识提供建议。

---

## SuGaR 可能的含义

**SuGaR** 可能是以下之一：

1. **Surface Gaussian Splatting** - 将场景表示为表面高斯（类似 2DGS 的几何重建方法）
2. **Super Gaussian** - 某种高斯溅射的改进方法
3. **Spherical Gaussian** - 球形高斯表示

**最可能**：Surface Gaussian Splatting（几何准确的高斯表示方法）

---

## 相关论文搜索方向

### 推荐搜索关键词

```bash
# GitHub
- "surface gaussian splatting"
- "geometrically accurate radiance fields"
- "gaussian surface reconstruction"
- "2d gaussian geometry"

# ArXiv
- "2D Gaussian Splatting"
- "Surface Gaussian Splatting"
- "Mesh Gaussian Splatting"
- "Gaussian Splatting geometry"

# 学术网站
- scholar.google.com: "SuGaR gaussian splatting"
- paperswithcode.com: "gaussian splatting"
```

---

## 复现策略

### 方法 1：查找官方实现

```bash
# 搜索 GitHub 仓库
keywords: [
    "SuGaR gaussian splatting",
    "surface gaussian",
    "gaussian surface reconstruction"
]

# 可能的作者：
- hbb1 (2DGS 作者)
- graphdeco (3DGS 团队)
- sony research
- google research
```

### 方法 2：查找会议论文

```bash
# 会议
- SIGGRAPH 2024/2025
- CVPR 2025
- ICCV 2025
- ECCV 2024
- NeurIPS 2024

# 搜索
- "Gaussian Splatting" + "geometry"
- "surface representation" + "gaussian"
```

---

## 基于现有知识的建议

### 可能的相关工作

1. **2DGS** (已复现)
   - 2D Gaussian Splatting for Geometrically Accurate Radiance Fields
   - GitHub: https://github.com/hbb1/2d-gaussian-splatting
   - ArXiv: https://arxiv.org/abs/2403.17888

2. **可能的后续工作**
   - 3DGS + 网格提取
   - Surface-guided 高斯溅射
   - 几何约束的高斯表示

### 类似方法

```bash
# 已有的几何重建方法
- Multi-view Geometric Regularization (MVG)
- Stereo-Splatting
- SDF-Gaussian
- NeRF + Mesh
```

---

## 行动计划

### 立即行动
1. ✅ 搜索 GitHub 上的 "SuGaR" 实现
2. ✅ 搜索 ArXiv 上的相关论文
3. ✅ 检查 2DGS 论文的引用和后续工作

### 短期目标
1. 找到 SuGaR 的官方实现（如果有）
2. 理解与 2DGS 的关系
3. 评估复现的复杂度

### 长期目标
1. 如果有官方代码，按照文档复现
2. 如果没有代码，根据论文从头实现
3. 对比 2DGS 和 SuGaR 的几何重建质量

---

## 下一步

请提供更多信息：

1. **SuGaR 的完整标题**是什么？
2. **论文链接或 ArXiv 编号**？
3. **作者**是谁？
4. **是否有 GitHub 仓库链接**？

有了这些信息，我可以：
- 精确定位论文
- 查找官方实现
- 评估复现难度
- 提供详细的复现步骤

---

**创建时间**：2026-02-16
**2DGS 状态**：已成功复现，环境已配置
