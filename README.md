# DiT Learning（扩散 Transformer 学习项目）

一个用于学习和复现 **DiT（Diffusion Transformer）** 的个人项目仓库。DiT 由 William Peebles 与谢赛宁在论文 *Scalable Diffusion Models with Transformers* 中提出，用 **Transformer** 替换传统扩散模型中的 **U-Net** 骨干网络，在潜在空间（latent space）中对图像块（patches）进行去噪生成。

本项目包含从基础扩散模型到条件生成、Flow Matching 等多种实现与实验 Notebook，适合作为理解 DiT 架构与扩散模型原理的学习材料。

---

## 项目结构

```
DiT/
├── src/
│   ├── condition_diffusion/       # 条件扩散模型相关实现
│   ├── data/                      # 数据加载与预处理
│   ├── diffusion/                 # 基础扩散模型模块
│   ├── flow_match/                # Flow Matching 实现
│   └── flow_match_dit/            # 结合 Flow Matching 的 DiT 实现
├── .gitignore
├── README.md                      # 原始文件（内容为空或未完善）
├── cond_dit_eval.ipynb            # 条件 DiT 评估 Notebook
├── condition_diffusion.ipynb      # 条件扩散模型实验
├── dit_eval.ipynb                 # DiT 模型评估
├── flow_dit_eval.ipynb            # Flow-DiT 评估
├── flow_eval.ipynb                # Flow Matching 评估
├── flow_match_dit_test.ipynb      # Flow Matching DiT 测试
├── flow_matching.ipynb            # Flow Matching 基础实验
├── simple_diffusion.ipynb         # 简单扩散模型入门实验
└── train.py                       # 训练脚本入口
```
---

## 核心概念：DiT 是什么

DiT（Diffusion Transformer）的核心思想是用 **Transformer** 替代扩散模型中常用的 **U-Net** 骨干：

- 将图像切成 **patch**，线性投影后输入 Transformer blocks
- 使用 **adaLN-Zero** 进行条件注入
- 在**潜在空间**中进行扩散去噪，而非像素空间
- 通过 **Gflops** 衡量前向复杂度，验证模型的可扩展性

DiT 是 **Sora** 等视频生成模型背后的重要技术基础之一，论文被 **ICCV 2023** 接收。

---

## 环境要求

- **Python** 3.8 及以上
- **PyTorch**（建议 1.12+）
- 支持 CUDA 的 GPU（用于训练与大规模生成）
- Jupyter Notebook / JupyterLab（运行 `.ipynb` 实验文件）
- 常用依赖：`numpy`、`matplotlib`、`einops`、`tqdm`

安装依赖：

```bash
pip install torch torchvision numpy matplotlib einops tqdm jupyter
```

---

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/lcdmiku/DiT.git
cd DiT
```

### 2. 运行 Notebook 实验

从最基础的扩散模型开始，逐步深入：

```bash
jupyter notebook simple_diffusion.ipynb
```

然后按顺序尝试：

1. `simple_diffusion.ipynb` — 扩散模型基础
2. `condition_diffusion.ipynb` — 条件扩散
3. `flow_matching.ipynb` — Flow Matching 基础
4. `flow_match_dit_test.ipynb` — Flow Matching + DiT 结合
5. `dit_eval.ipynb` / `cond_dit_eval.ipynb` — DiT 评估

### 3. 训练（如 train.py 可用）

```bash
python train.py
```

> `train.py` 的具体参数与功能需查看源码确认。

---

## 模块说明

| 模块 / 文件 | 用途 |
|-------------|------|
| `src/diffusion/` | 基础扩散过程（前向加噪、反向去噪）实现 |
| `src/condition_diffusion/` | 带条件（如类别标签）的扩散模型 |
| `src/flow_match/` | Flow Matching 训练与采样逻辑 |
| `src/flow_match_dit/` | Flow Matching 与 DiT 架构的结合 |
| `src/data/` | 数据集加载与预处理工具 |
| `*.ipynb` | 各阶段的实验与评估 Notebook |

---

## 相关论文与参考

- **DiT 原论文**：William Peebles, Saining Xie. *Scalable Diffusion Models with Transformers*. ICCV 2023. [arXiv:2212.09748](https://arxiv.org/abs/2212.09748)
- **官方实现**：[facebookresearch/DiT](https://github.com/facebookresearch/DiT)
- **Flow Matching**：Lipman et al. *Flow Matching for Generative Modeling*

---

## 注意事项

- 本仓库为**个人学习项目**，代码与 Notebook 可能处于持续修改状态，不保证完整性与可复现性。
- 部分 Notebook 可能依赖本地数据集或预训练权重，运行前请检查路径配置。

---

## 许可证

本项目未声明许可证文件。如需使用、修改或分发，请先联系仓库作者获取授权。

## 作者

- **lcdmiku** — [GitHub 主页](https://github.com/lcdmiku)

---
