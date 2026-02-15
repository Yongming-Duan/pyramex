# PyRamEx - PyPI发布准备清单

**状态：** 准备中（v0.1.0-beta）

---

## ✅ 已完成

### 1. 代码准备
- [x] 核心功能实现
- [x] 模块结构完整
- [x] 类型提示
- [x] 文档字符串

### 2. 项目配置
- [x] setup.py
- [x] pyproject.toml
- [x] requirements.txt
- [x] .gitignore

### 3. 文档
- [x] README.md
- [x] CONTRIBUTING.md
- [x] LICENSE (MIT)
- [x] NOTICE.md

### 4. CI/CD
- [x] GitHub Actions工作流
- [x] 自动测试配置
- [x] 代码覆盖率配置

---

## 🔲 待完成

### 1. PyPI账户准备
- [ ] 注册PyPI账户
- [ ] 启用双因素认证（2FA）
- [ ] 创建API token
- [ ] 保存token（只显示一次）

**PyPI注册：** https://pypi.org/account/register/

**创建API Token：**
1. 访问：https://pypi.org/manage/account/token/
2. Token名称：pyramex-token
3. 范围：Entire account（或仅pyramex）
4. 创建并复制token

**添加到GitHub Secrets：**
1. 访问：https://github.com/YOUR_USERNAME/pyramex/settings/secrets/actions
2. Name: `PYPI_API_TOKEN`
3. Value: 粘贴token
4. 点击"Add secret"

### 2. 单元测试
- [ ] tests/test_core.py
- [ ] tests/test_io.py
- [ ] tests/test_preprocessing.py
- [ ] tests/test_qc.py
- [ ] tests/test_features.py
- [ ] tests/test_ml.py
- [ ] tests/test_visualization.py

**目标覆盖率：** >= 80%

### 3. 版本号配置
- [ ] 在pyproject.toml中配置动态版本
- [ ] 创建_version.py文件
- [ ] 测试版本号显示

### 4. 包构建测试
```bash
# 安装构建工具
pip install build twine

# 构建包
python -m build

# 检查包
twine check dist/*

# 测试安装（本地）
pip install dist/pyramex-0.1.0b0.tar.gz
python -c "from pyramex import Ramanome; print(Ramanome.__version__)"
```

### 5. TestPyPI发布（先测试）
```bash
# 注册TestPyPI：https://test.pypi.org/account/register/

# 发布到TestPyPI
twine upload --repository testpypi dist/*

# 测试安装
pip install --index-url https://test.pypi.org/simple/ pyramex
```

### 6. 正式PyPI发布
```bash
# 确保版本号正确
# 构建包
python -m build

# 发布到PyPI
twine upload dist/*

# 验证
pip install pyramex
python -c "from pyramex import Ramanome; print('Success!')"
```

### 7. GitHub Release
1. 访问：https://github.com/YOUR_USERNAME/pyramex/releases/new
2. Tag: `v0.1.0b0`
3. Title: `PyRamEx v0.1.0-beta`
4. Description: 发布说明
5. 发布

---

## 📋 发布前检查清单

### 代码质量
- [ ] 所有测试通过
- [ ] 代码格式化
- [ ] 类型检查
- [ ] 无lint警告

### 文档完整性
- [ ] README清晰
- [ ] API文档完整
- [ ] 安装说明正确
- [ ] 示例代码可运行

### 版本管理
- [ ] 版本号正确（v0.1.0b0）
- [ ] CHANGELOG更新
- [ ] Release notes准备

### 安全性
- [ ] 无敏感信息泄露
- [ ] 依赖项安全检查
- [ ] License正确

---

## 🎯 发布时间线

**Phase 1: 准备（本周）**
- Day 1: PyPI账户 + 单元测试
- Day 2-3: 完善测试（覆盖率>=80%）
- Day 4: TestPyPI发布测试
- Day 5: 正式PyPI发布

**Phase 2: 验证（下周）**
- 测试安装
- 收集反馈
- 修复bug

**Phase 3: 稳定版（v0.2.0）**
- 高级功能
- 性能优化
- 正式发布

---

## 📝 版本号策略

```
v0.1.0a0 - Alpha（当前）
v0.1.0b0 - Beta（目标）
v0.2.0 - Feature complete
v1.0.0 - Stable release
```

---

**详细指南已保存！准备好后我将协助完成发布流程。**