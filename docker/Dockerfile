FROM docker.1ms.run/library/python:3.13-slim

WORKDIR /app

# 安装 supervisor
RUN apt-get update && apt-get install -y --no-install-recommends \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件，利用 Docker 层缓存
COPY requirements.txt .

# 安装 Python 依赖（清华pip源加速）
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制项目代码
COPY . .

# 创建必要目录并放开临时目录权限
RUN mkdir -p /app/kb_docs /app/temp_summary && chmod 777 /app/temp_summary

# supervisor 配置
COPY docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

EXPOSE 7862 7863

# 仅保留一条启动命令
CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/conf.d/supervisord.conf"]