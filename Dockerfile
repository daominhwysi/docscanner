FROM python:3.10-slim
RUN apt-get update && apt-get install -y supervisor && \
    mkdir -p /var/log/supervisor
WORKDIR /src

COPY . /src

RUN pip install --no-cache-dir -r requirements.txt

COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Mở cổng cho FastAPI
EXPOSE 8000

# 🔥 Chạy supervisor (thay vì uvicorn)
CMD ["/usr/bin/supervisord"]