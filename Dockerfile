FROM python:3.10-slim
RUN apt-get update && apt-get install -y supervisor && \
    mkdir -p /var/log/supervisor
WORKDIR /src

COPY . /src

RUN pip install --no-cache-dir -r requirements.txt

# Cài đặt Supervisor
RUN mkdir -p /etc/supervisor

# Copy file config
COPY supervisord.conf /etc/supervisor/supervisord.conf

# Mở cổng cho FastAPI
EXPOSE 8000

# 🔥 Chạy supervisor (thay vì uvicorn)
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/supervisord.conf"]