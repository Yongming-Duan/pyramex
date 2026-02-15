# 发布审核清单

**项目：** PyRamEx
**版本：** v0.1.0-alpha
**最后更新：** 2026-02-15 21:05

---

## 📋 发布前检查清单

### 1. 代码审核
- [ ] 代码格式化（Black）
- [ ] 类型检查（mypy）
- [ ] Linting（flake8）
- [ ] 所有测试通过
- [ ] 测试覆盖率 >= 80%

### 2. 文档审核
- [ ] README.md准确完整
- [ ] API文档已更新
- [ ] 示例代码可运行
- [ ] 安装说明正确

### 3. 敏感信息审核 ⚠️ **重点**
- [ ] 运行敏感信息检查脚本
  ```bash
  ./scripts/check_sensitive_info.sh
  ```
- [ ] 检查Git diff中的敏感信息
  ```bash
  git diff --cached
  ```
- [ ] 确认没有Token、密码、API密钥
- [ ] 确认没有硬编码路径
- [ ] 确认没有个人隐私信息

### 4. 安全审核
- [ ] 依赖包版本检查
- [ ] 已知漏洞扫描
- [ ] 许可证兼容性
- [ ] 私密信息已加密

### 5. 功能审核
- [ ] 核心功能可用
- [ ] 边界情况处理
- [ ] 错误处理完整
- [ ] 性能可接受

---

## 🔍 敏感信息审核详情

### L1级 - 公开信息（✅ 可发布）
**项目信息：**
- [x] 仓库名称: pyramex
- [x] GitHub用户名: Yongming-Duan
- [x] 许可证: MIT
- [x] 项目描述

**技术信息：**
- [x] Python版本要求
- [x] 依赖包列表
- [x] 功能特性

### L2级 - 内部信息（⚠️ 需审核）
**开发信息：**
- [ ] 代码架构文档
- [ ] 性能基准数据
- [ ] 开发计划

**审核要点：**
- 是否包含内部API？
- 是否暴露实现细节？
- 是否需要脱敏？

### L3级 - 敏感信息（🔒 严禁发布）
**认证凭证：**
- [x] GitHub Token: **已撤销**
- [ ] PyPI Token
- [ ] 其他API密钥

**服务器信息：**
- [ ] 服务器IP
- [ ] 部署路径
- [ ] 配置文件

**个人信息：**
- [ ] 真实邮箱（公开邮箱除外）
- [ ] 手机号码
- [ ] 私人地址

---

## 🚨 常见敏感信息泄露场景

### 场景1：Git历史记录
**风险：** 敏感信息已提交到历史

**检查：**
```bash
# 搜索所有提交中的敏感信息
git log -p --all -S "ghp_" -- "*.py" "*.md"
```

**修复：**
```bash
# 使用git-filter-repo清除历史
git filter-repo --invert-paths SECRET_FILE
```

### 场景2：示例代码
**风险：** 示例代码包含真实凭证

**检查：**
```bash
grep -r "token\|password\|api_key" examples/
```

**修复：**
使用占位符：
```python
# ❌ 错误
API_TOKEN = "ghp_xxx"

# ✅ 正确
API_TOKEN = os.getenv("API_TOKEN", "YOUR_TOKEN_HERE")
```

### 场景3：文档中的示例
**风险：** 文档包含真实路径、URL

**检查：**
```bash
grep -r "/home/\|http://" README.md docs/
```

**修复：**
使用通用示例：
```markdown
# ❌ 错误
Load data from /home/user/data/spectra/

# ✅ 正确
Load data from /path/to/spectra/
```

---

## 📝 发布审核流程

### Step 1: 自动检查
```bash
# 运行敏感信息检查
./scripts/check_sensitive_info.sh

# 运行测试
pytest

# 运行linting
black --check pyramex/
flake8 pyramex/
mypy pyramex/
```

### Step 2: 手动审核
1. 审查所有变更文件
   ```bash
   git status
   git diff
   ```

2. 检查新增文件
   ```bash
   git diff --cached --name-only
   ```

3. 查看最终提交
   ```bash
   git diff HEAD~1
   ```

### Step 3: 测试发布
1. 发布到TestPyPI
   ```bash
   twine upload --repository testpypi dist/*
   ```

2. 测试安装
   ```bash
   pip install --index-url https://test.pypi.org/simple/ pyramex
   ```

3. 功能测试
   ```bash
   python -c "from pyramex import Ramanome; print('OK')"
   ```

### Step 4: 正式发布
1. 确认所有检查通过
2. 创建Git Tag
   ```bash
   git tag v0.1.0b0
   git push origin v0.1.0b0
   ```

3. 发布到PyPI
   ```bash
   twine upload dist/*
   ```

4. 验证
   ```bash
   pip install pyramex
   ```

---

## ✅ 发布后检查

- [ ] PyPI页面显示正确
- [ ] GitHub Release创建
- [ ] 文档更新
- [ ] CHANGELOG更新
- [ ] 撤销临时Token

---

## 📞 审核疑问

**发现敏感信息怎么办？**
1. 立即停止发布
2. 删除或脱敏敏感信息
3. 如果已提交，清理Git历史
4. 重新审核

**审核不通过怎么办？**
1. 修复所有问题
2. 重新运行检查
3. 获取二次审核
4. 确认后发布

---

**审核人：** 小龙虾1号 🦞
**最后更新：** 2026-02-15 21:05
**下次审核：** 每次发布前

---

*本文档必须严格遵守，确保信息安全*
