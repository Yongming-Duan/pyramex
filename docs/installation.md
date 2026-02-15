# PyRamEx 安装指南

PyRamEx是一个用于拉曼光谱分析的Python工具包，提供ML/DL友好的API。

## 系统要求

- Python 3.8 或更高版本
- 操作系统：Windows、macOS 或 Linux

## 安装方法

### 方法1：使用pip安装（推荐）

```bash
pip install pyramex
```

### 方法2：从源码安装

```bash
# 克隆仓库
git clone https://github.com/Yongming-Duan/pyramex.git
cd pyramex

# 安装
pip install -e .
```

### 方法3：从GitHub下载

```bash
# 下载并解压
wget https://github.com/Yongming-Duan/pyramex/archive/refs/heads/main.zip
unzip main.zip
cd pyramex-main

# 安装
pip install -e .
```

## 依赖项

### 核心依赖

PyRamEx的核心功能需要以下依赖：

```
numpy>=1.20.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
```

### 可选依赖

某些高级功能需要额外的包：

#### 机器学习集成

```bash
# PyTorch支持
pip install torch

# TensorFlow支持
pip install tensorflow
```

#### 高级降维

```bash
# UMAP降维
pip install umap-learn
```

#### 交互式可视化

```bash
# Plotly交互式图表
pip install plotly
```

## 验证安装

安装完成后，可以通过以下命令验证：

```bash
python -c "import pyramex; print(pyramex.__version__)"
```

应该输出版本号：`0.1.0`

或者运行测试套件：

```bash
python -m pytest pyramex/tests/
```

## 常见问题

### Q1: 安装失败提示缺少编译器

某些依赖包（如numpy、scipy）可能需要C编译器。

**解决方案：**

**Ubuntu/Debian:**
```bash
sudo apt-get install python3-dev build-essential
```

**macOS:**
```bash
xcode-select --install
```

**Windows:**
安装Visual Studio Build Tools或使用预编译的wheel包。

### Q2: PyTorch或TensorFlow安装问题

PyTorch和TensorFlow需要根据您的系统配置（CPU/GUDA）选择特定版本。

**PyTorch安装：**
访问 [PyTorch官网](https://pytorch.org/get-started/locally/) 获取适合您系统的安装命令。

**TensorFlow安装：**
```bash
# CPU版本
pip install tensorflow-cpu

# GPU版本（需要CUDA）
pip install tensorflow
```

### Q3: 导入错误

如果遇到导入错误，尝试：

```bash
# 重新安装
pip uninstall pyramex
pip install pyramex

# 或从源码安装
cd pyramex
pip install -e . --force-reinstall
```

### Q4: 版本冲突

如果遇到依赖版本冲突，可以使用虚拟环境：

```bash
# 创建虚拟环境
python -m venv pyramex_env
source pyramex_env/bin/activate  # Linux/macOS
# 或
pyramex_env\\Scripts\\activate  # Windows

# 安装PyRamEx
pip install pyramex
```

## 开发者安装

如果您想参与PyRamEx的开发：

```bash
# 克隆仓库
git clone https://github.com/Yongming-Duan/pyramex.git
cd pyramex

# 安装开发依赖
pip install -e ".[dev]"

# 安装pre-commit钩子
pre-commit install
```

开发依赖包括：
- pytest - 测试框架
- pytest-cov - 代码覆盖率
- black - 代码格式化
- flake8 - 代码检查
- mypy - 类型检查

## Docker安装

使用Docker可以避免环境配置问题：

```bash
# 拉取镜像（如果提供）
docker pull ghcr.io/yongming-duan/pyramex:latest

# 或使用Dockerfile构建
docker build -t pyramex .
```

运行容器：

```bash
docker run -it pyramex python
```

## 下一步

安装完成后，请查看：

- [快速开始教程](tutorial.md) - 学习基本用法
- [用户指南](user_guide.md) - 了解高级功能
- [API参考](api.md) - 查看完整API文档

## 获取帮助

如果遇到问题：

1. 查看[常见问题](#常见问题)
2. 搜索[GitHub Issues](https://github.com/Yongming-Duan/pyramex/issues)
3. 创建新的Issue描述您的问题

---

**版本：** 0.1.0-beta
**最后更新：** 2026-02-15
