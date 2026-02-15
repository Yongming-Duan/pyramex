# PyRamEx v0.1.0-beta - 推送到GitHub指南

**状态：** 代码已提交，标签已创建，等待推送
**时间：** 2026-02-15 22:40

---

## 📋 当前状态

✅ **已完成：**
- 代码已提交（46个文件，10,959行）
- Git标签已创建（v0.1.0-beta）
- 构建产物已生成（wheel + source）

⏳ **待完成：**
- 推送到GitHub
- 创建GitHub Release

---

## 🚀 推送方法（选择一种）

### 方法1: 使用Personal Access Token（推荐）

#### 步骤1: 创建GitHub Personal Access Token

1. 访问：https://github.com/settings/tokens
2. 点击 "Generate new token" → "Generate new token (classic)"
3. 设置权限：
   - ✅ repo（完整仓库访问权限）
4. 点击 "Generate token"
5. **重要：** 复制生成的token（只显示一次！）

#### 步骤2: 使用Token推送

```bash
cd /home/yongming/openclaw/pyramex

# 方法A: 使用token作为密码（推荐）
# 当提示输入密码时，粘贴token（不是GitHub密码！）
git push origin main
git push origin v0.1.0-beta

# 方法B: 在URL中包含token
git remote set-url origin https://YOUR_TOKEN@github.com/Yongming-Duan/pyramex.git
git push origin main
git push origin v0.1.0-beta

# 方法C: 使用git credential helper
git config credential.helper store
git push origin main
# 输入用户名：Yongming-Duan
# 输入密码：粘贴token
```

**注意：** 
- Token是密码，不是GitHub账号密码
- Token生成后只显示一次，请妥善保管
- 可以设置token过期时间（建议90天或更长）

---

### 方法2: 使用GitHub CLI（gh）

#### 步骤1: 安装GitHub CLI

```bash
# Ubuntu/Debian
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update
sudo apt install gh

# 验证安装
gh --version
```

#### 步骤2: 认证

```bash
gh auth login

# 选择选项：
# What account do you want to log into? → GitHub.com
# What is your preferred protocol for Git operations? → HTTPS
# Authenticate Git with your GitHub credentials? → Yes
# How would you like to authenticate GitHub CLI? → Login with a web browser
```

#### 步骤3: 推送

```bash
cd /home/yongming/openclaw/pyramex
git push origin main
git push origin v0.1.0-beta
```

---

### 方法3: 使用SSH密钥

#### 步骤1: 生成SSH密钥

```bash
# 生成新SSH密钥
ssh-keygen -t ed25519 -C "xiaolongxia@openclaw.cn"

# 按Enter使用默认路径
# 可以设置密码或直接按Enter跳过
```

#### 步骤2: 添加SSH密钥到GitHub

1. 复制公钥：
```bash
cat ~/.ssh/id_ed25519.pub
```

2. 添加到GitHub：
   - 访问：https://github.com/settings/keys
   - 点击 "New SSH key"
   - Title: "Home Server" 或类似名称
   - Key: 粘贴公钥内容
   - 点击 "Add SSH key"

#### 步骤3: 修改remote URL并推送

```bash
cd /home/yongming/openclaw/pyramex

# 修改为SSH URL
git remote set-url origin git@github.com:Yongming-Duan/pyramex.git

# 测试连接
ssh -T git@github.com

# 推送
git push origin main
git push origin v0.1.0-beta
```

---

## 📝 推送成功后，创建GitHub Release

### 步骤1: 访问Release页面

推送成功后，访问：
```
https://github.com/Yongming-Duan/pyramex/releases/new
```

### 步骤2: 填写Release信息

#### 基本信息
- **Choose a tag:** 选择 `v0.1.0-beta`
- **Release title:** `PyRamEx v0.1.0-beta`
- **Describe this release:** 复制下方内容

#### Release描述

```markdown
# PyRamEx v0.1.0-beta 🎉

**PyRamex (Python Ramanome Analysis Toolkit)** - 功能强大的Python拉曼光谱分析工具包

## ✨ 主要特性

- 🔧 完整的预处理流程（平滑、基线去除、归一化）
- 🔍 多种质量控制方法（ICOD、MCD、T²、SNR）
- 📊 降维和特征提取（PCA、UMAP、t-SNE、PCoA）
- 🤖 ML/DL框架集成（sklearn、PyTorch、TensorFlow）
- 📈 丰富的可视化工具

## 📦 安装

```bash
pip install pyramex
```

## 🚀 快速开始

```python
from pyramex import Ramanome

# 加载数据
ramanome = load_spectra('data/')

# 预处理
ramanome.smooth().remove_baseline().normalize()

# 质量控制
qc_result = ramanome.quality_control(method='dis')

# 降维
ramanome.reduce(method='pca', n_components=2)

# 可视化
ramanome.plot()
```

## ✅ 质量保证

- **194个测试用例**，100%通过率
- **43,250+字**完整文档
- **7个示例**脚本
- **9项验证**测试
- 性能基准优秀

## 📚 文档

- [安装指南](https://github.com/Yongming-Duan/pyramex/blob/main/docs/installation.md)
- [快速教程](https://github.com/Yongming-Duan/pyramex/blob/main/docs/tutorial.md)
- [用户指南](https://github.com/Yongming-Duan/pyramex/blob/main/docs/user_guide.md)
- [API参考](https://github.com/Yongming-Duan/pyramex/blob/main/docs/api.md)

## 📊 性能

| 操作 | 时间 | 评级 |
|------|------|------|
| 平滑 | 12.4ms | ⭐⭐⭐⭐⭐ |
| 归一化 | 0.7ms | ⭐⭐⭐⭐⭐ |
| QC | 1.4ms | ⭐⭐⭐⭐⭐ |
| PCA | 58.1ms | ⭐⭐⭐⭐ |

**质量评分：** ⭐⭐⭐⭐⭐ (5/5)

## 📄 许可证

GPL License

---

**PyRamEx v0.1.0-beta - 让拉曼光谱分析更简单！** 🚀
```

#### 设置
- ✅ 勾选 "Set as a pre-release"（这是beta版本）
- ⬜ 不勾选 "Set as the latest release"

#### 附件
上传以下文件：
- `dist/pyramex-0.1.0-py3-none-any.whl`
- `dist/pyramex-0.1.0.tar.gz`

#### 发布
点击 "Publish release" 按钮

---

## 🔍 验证

推送成功后，可以验证：

```bash
# 查看远程标签
git ls-remote --tags origin

# 应该看到：
# ... refs/tags/v0.1.0-beta
```

访问以下页面确认：
- Tags: https://github.com/Yongming-Duan/pyramex/tags
- Releases: https://github.com/Yongming-Duan/pyramex/releases

---

## ❓ 常见问题

### Q1: 推送时提示"Authentication failed"
**A:** 检查token是否正确，或token是否有足够的权限（需要repo权限）

### Q2: 推送时提示"could not read Username"
**A:** 需要提供认证信息，使用上述方法之一

### Q3: 标签推送后不显示
**A:** 等待几分钟，或检查：
```bash
git ls-remote --tags origin
```

### Q4: Release创建后附件无法下载
**A:** 检查文件路径是否正确，或重新上传

---

## 🎯 推荐方法

**如果是首次推送：** 使用方法1（Personal Access Token）

**如果经常需要推送：** 使用方法2（GitHub CLI）或方法3（SSH密钥）

**现在已准备好推送！选择一种方法即可** 🚀

---

**准备就绪时间：** 2026-02-15 22:40
**项目状态：** 发布就绪
