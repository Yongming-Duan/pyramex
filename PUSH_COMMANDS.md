# PyRamEx - GitHub推送命令

**GitHub用户名：** Yongming-Duan
**仓库URL：** https://github.com/Yongming-Duan/pyramex
**准备时间：** 2026-02-15 20:55

---

## 📋 推送命令

### 前提条件

**重要：** 请先在GitHub网页创建仓库！

1. 访问：https://github.com/new
2. Repository name: `pyramex`
3. Description: `A Python Ramanome Analysis Toolkit for ML/DL-friendly analysis`
4. ☑️ Public
5. **不要**勾选"Add a README file"
6. 点击"Create repository"

---

## 🚀 推送命令

```bash
cd /home/yongming/openclaw/pyramex

# 远程仓库已配置
git remote -v
# origin	https://github.com/Yongming-Duan/pyramex.git (fetch)
# origin	https://github.com/Yongming-Duan/pyramex.git (push)

# 分支已重命名
git branch
# main

# 推送命令
git push -u origin main
```

---

## 🔑 认证说明

### 使用Personal Access Token（推荐）

**推送时会要求：**
- Username: `Yongming-Duan`
- Password: **你的Personal Access Token**（不是密码！）

**如果没有Token，创建一个：**
1. 访问：https://github.com/settings/tokens
2. 点击 "Generate new token" → "Generate new token (classic)"
3. 勾选权限：
   - ✅ **repo**（全部子项）
   - ✅ **workflow**（全部子项）
4. 设置过期时间（如90天）
5. 点击"Generate token"
6. **立即复制token**（只显示一次！）

**推送时：**
```
Username: Yongming-Duan
Password: <粘贴token>
```

### 使用SSH（如果已配置）

如果已配置SSH密钥，可以切换到SSH方式：

```bash
cd /home/yongming/openclaw/pyramex

# 删除HTTPS remote
git remote remove origin

# 添加SSH remote
git remote add origin git@github.com:Yongming-Duan/pyramex.git

# 推送
git push -u origin main
```

---

## ✅ 推送后自动触发

推送成功后，GitHub Actions会自动运行：

**自动执行：**
- ✅ 单元测试（Python 3.8, 3.9, 3.10, 3.11）
- ✅ 代码覆盖率检查
- ✅ 代码格式检查
- ✅ 包构建测试

**查看Actions：**
```
https://github.com/Yongming-Duan/pyramex/actions
```

---

## 🎯 验证清单

推送成功后检查：

- [ ] 代码在GitHub上可见
- [ ] README.md正确显示
- [ ] Actions标签页有工作流运行
- [ ] CI/CD工作流正在执行

---

## 📝 完整流程

1. **创建GitHub仓库**（网页）
   ✅ https://github.com/new
   ✅ Repository name: pyramex
   ✅ Public
   ✅ 不要初始化README

2. **推送代码**（命令行）
   ```bash
   cd /home/yongming/openclaw/pyramex
   git push -u origin main
   ```

3. **验证CI/CD**（网页）
   ✅ 访问Actions页面
   ✅ 查看工作流运行

---

**准备好后，我可以执行推送命令！**

或者你可以手动执行上面的命令。

---

**当前状态：**
✅ Remote已配置：origin → https://github.com/Yongming-Duan/pyramex.git
✅ 分支已重命名：master → main
⏳ 等待推送

**请在GitHub网页创建仓库后告诉我，我立即推送！**
