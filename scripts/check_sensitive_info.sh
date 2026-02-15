#!/bin/bash
# 敏感信息检查脚本
# 用途：在发布前检查代码和文档中是否包含敏感信息

echo "🔍 PyRamEx 发布前敏感信息检查"
echo "================================"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查计数
ERRORS=0
WARNINGS=0

echo ""
echo "📋 检查项："
echo ""

# 1. 检查GitHub Token
echo "1️⃣  检查GitHub Token..."
if grep -r "ghp_" pyramex/ --include="*.py" --include="*.md" --include="*.txt" 2>/dev/null; then
    echo -e "${RED}❌ 发现GitHub Token！${NC}"
    ERRORS=$((ERRORS + 1))
else
    echo -e "${GREEN}✅ 未发现GitHub Token${NC}"
fi

# 2. 检查密码
echo ""
echo "2️⃣  检查密码关键词..."
if grep -r "password\|passwd\|pwd" pyramex/ --include="*.py" --include="*.md" 2>/dev/null | grep -v "password_hash\|password_requirements"; then
    echo -e "${YELLOW}⚠️  发现可能包含密码的内容${NC}"
    grep -r "password\|passwd\|pwd" pyramex/ --include="*.py" --include="*.md" 2>/dev/null | grep -v "password_hash"
    WARNINGS=$((WARNINGS + 1))
else
    echo -e "${GREEN}✅ 未发现密码关键词${NC}"
fi

# 3. 检查API密钥
echo ""
echo "3️⃣  检查API密钥..."
if grep -r "api_key\|apikey\|api-key" pyramex/ --include="*.py" --include="*.md" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  发现API密钥关键词${NC}"
    grep -r "api_key\|apikey\|api-key" pyramex/ --include="*.py" --include="*.md" 2>/dev/null
    WARNINGS=$((WARNINGS + 1))
else
    echo -e "${GREEN}✅ 未发现API密钥关键词${NC}"
fi

# 4. 检查硬编码路径
echo ""
echo "4️⃣  检查硬编码路径..."
if grep -r "/home/yongming\|/root/\|~/" pyramex/ --include="*.py" 2>/dev/null | grep -v "README\|docs\|examples"; then
    echo -e "${YELLOW}⚠️  发现硬编码路径${NC}"
    grep -r "/home/yongming\|/root/\|~/" pyramex/ --include="*.py" 2>/dev/null | grep -v "README\|docs\|examples" | head -5
    WARNINGS=$((WARNINGS + 1))
else
    echo -e "${GREEN}✅ 未发现硬编码路径${NC}"
fi

# 5. 检查私钥文件
echo ""
echo "5️⃣  检查私钥文件..."
if find pyramex/ -name "*.pem" -o -name "*.key" -o -name "*secret*" 2>/dev/null | grep -v ".git"; then
    echo -e "${RED}❌ 发现私钥文件！${NC}"
    find pyramex/ -name "*.pem" -o -name "*.key" -o -name "*secret*" 2>/dev/null | grep -v ".git"
    ERRORS=$((ERRORS + 1))
else
    echo -e "${GREEN}✅ 未发现私钥文件${NC}"
fi

# 6. 检查环境变量引用
echo ""
echo "6️⃣  检查环境变量引用..."
if grep -r "os.getenv\|os.environ" pyramex/ --include="*.py" 2>/dev/null; then
    echo -e "${GREEN}✅ 使用环境变量（安全）${NC}"
else
    echo -e "${YELLOW}⚠️  未使用环境变量${NC}"
fi

# 7. 检查IP地址
echo ""
echo "7️⃣  检查IP地址..."
if grep -rE "\b[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\b" pyramex/ --include="*.py" --include="*.md" 2>/dev/null | grep -v "127.0.0.1\|0.0.0.0"; then
    echo -e "${YELLOW}⚠️  发现IP地址${NC}"
    grep -rE "\b[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\b" pyramex/ --include="*.py" --include="*.md" 2>/dev/null | grep -v "127.0.0.1\|0.0.0.0" | head -3
    WARNINGS=$((WARNINGS + 1))
else
    echo -e "${GREEN}✅ 未发现IP地址${NC}"
fi

# 8. 检查邮箱地址
echo ""
echo "8️⃣  检查邮箱地址..."
if grep -rE "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}" pyramex/ --include="*.py" --include="*.md" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  发现邮箱地址${NC}"
    grep -rE "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}" pyramex/ --include="*.py" --include="*.md" 2>/dev/null | head -3
    echo -e "${YELLOW}   （请确认是否可以公开）${NC}"
    WARNINGS=$((WARNINGS + 1))
else
    echo -e "${GREEN}✅ 未发现邮箱地址${NC}"
fi

# 总结
echo ""
echo "================================"
echo "📊 检查结果："
echo -e "错误：${RED}${ERRORS}${NC}"
echo -e "警告：${YELLOW}${WARNINGS}${NC}"
echo ""

if [ $ERRORS -gt 0 ]; then
    echo -e "${RED}❌ 检查未通过！发现 ${ERRORS} 个错误${NC}"
    echo "请修复后重试"
    exit 1
elif [ $WARNINGS -gt 0 ]; then
    echo -e "${YELLOW}⚠️  发现 ${WARNINGS} 个警告，请确认${NC}"
    echo "如果确认安全，可以继续发布"
    exit 0
else
    echo -e "${GREEN}✅ 检查通过！可以安全发布${NC}"
    exit 0
fi
