FROM python:3.13-slim

WORKDIR /app

# 安装 supervisor（管理多进程）
RUN apt-get update && apt-get install -y --no-install-recommends \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖文件，利用 Docker 层缓存
COPY requirements.txt .

# 安装 Python 依赖（国内镜像加速）
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制项目代码
COPY . .

# 创建必要的目录
RUN mkdir -p /app/kb_docs /app/temp_summary

# supervisor 配置
COPY docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

EXPOSE 7862 7863

CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/conf.d/supervisord.conf"]