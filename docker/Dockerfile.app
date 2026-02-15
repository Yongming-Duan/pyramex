# PyRamEx主应用Dockerfile
FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 安装额外的ML依赖
RUN pip install --no-cache-dir \
    fastapi[all] \
    uvicorn[standard] \
    sqlalchemy \
    psycopg2-binary \
    redis \
    python-multipart

# 复制应用代码
COPY . .

# 创建日志目录
RUN mkdir -p /logs

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# 启动命令
CMD ["uvicorn", "pyramex.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
