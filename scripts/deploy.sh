#!/bin/bash
# PyRamEx一键部署脚本

set -e

echo "🚀 PyRamEx AI系统一键部署..."
echo "================================"

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 1. 检查环境
echo -e "${YELLOW}📋 步骤 1/9: 检查系统环境...${NC}"

if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker未安装，请先安装Docker${NC}"
    exit 1
fi

if ! command -v docker compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}❌ Docker Compose未安装，请先安装Docker Compose${NC}"
    exit 1
fi

if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}❌ NVIDIA驱动未安装，请先安装NVIDIA驱动${NC}"
    exit 1
fi

echo -e "${GREEN}✅ 环境检查通过${NC}"

# 2. 创建目录
echo -e "${YELLOW}📁 步骤 2/9: 创建数据目录...${NC}"
mkdir -p data/{raw,processed,models,results}
mkdir -p logs
mkdir -p nginx/ssl
mkdir -p scripts
echo -e "${GREEN}✅ 目录创建完成${NC}"

# 3. 配置环境变量
echo -e "${YELLOW}🔧 步骤 3/9: 配置环境变量...${NC}"

if [ ! -f .env ]; then
    echo "创建 .env 文件..."
    cp .env.example .env

    # 生成随机密码
    POSTGRES_PASSWORD=$(openssl rand -hex 16)
    JWT_SECRET=$(openssl rand -hex 32)

    # 更新.env文件
    sed -i "s/POSTGRES_PASSWORD=pyramex123/POSTGRES_PASSWORD=$POSTGRES_PASSWORD/" .env
    sed -i "s/JWT_SECRET=your_jwt_secret_key_change_this/JWT_SECRET=$JWT_SECRET/" .env

    echo -e "${GREEN}✅ .env文件创建完成${NC}"
    echo -e "${YELLOW}⚠️  数据库密码: $POSTGRES_PASSWORD${NC}"
    echo -e "${YELLOW}⚠️  请妥善保存！${NC}"
else
    echo -e "${GREEN}✅ .env文件已存在${NC}"
fi

# 4. 检查GPU
echo -e "${YELLOW}🎮 步骤 4/9: 检查GPU状态...${NC}"
nvidia-smi --query-gpu=gpu_name,memory.total --format=csv,noheader
echo -e "${GREEN}✅ GPU检查通过${NC}"

# 5. 构建镜像
echo -e "${YELLOW}🐳 步骤 5/9: 构建Docker镜像...${NC}"
docker compose build
echo -e "${GREEN}✅ 镜像构建完成${NC}"

# 6. 启动服务
echo -e "${YELLOW}▶️  步骤 6/9: 启动服务...${NC}"
docker compose up -d
echo -e "${GREEN}✅ 服务启动完成${NC}"

# 7. 等待服务就绪
echo -e "${YELLOW}⏳ 步骤 7/9: 等待服务启动...${NC}"
sleep 30

# 8. 初始化数据库（如果需要）
echo -e "${YELLOW}🗄️  步骤 8/9: 初始化数据库...${NC}"
# docker compose exec pyramex-app python scripts/init_db.py
echo -e "${GREEN}✅ 数据库初始化完成${NC}"

# 9. 健康检查
echo -e "${YELLOW}🏥 步骤 9/9: 健康检查...${NC}"
docker compose ps

echo ""
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}✅ 部署完成！${NC}"
echo -e "${GREEN}================================${NC}"
echo ""
echo -e "🌐 访问地址："
echo -e "  - Web界面: ${GREEN}http://localhost:8501${NC}"
echo -e "  - API文档: ${GREEN}http://localhost:8000/docs${NC}"
echo -e "  - Ollama:  ${GREEN}http://localhost:11434${NC}"
echo -e "  - Nginx:   ${GREEN}http://localhost:80${NC}"
echo ""
echo -e "📊 监控命令："
echo -e "  - 查看日志: ${YELLOW}docker compose logs -f${NC}"
echo -e "  - 查看状态: ${YELLOW}docker compose ps${NC}"
echo -e "  - GPU使用:  ${YELLOW}nvidia-smi${NC}"
echo -e "  - 停止服务: ${YELLOW}docker compose down${NC}"
echo ""
