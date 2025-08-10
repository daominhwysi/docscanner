# lib/redis_client.py
import os
import redis.asyncio as redis

class RedisClient:
    def __init__(self):
        self.client = None
        self.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.redis_port = int(os.getenv("REDIS_PORT", 6379))
        self.redis_password = os.getenv('REDIS_PASSWORD')
    def get_connection(self, decode_responses=False):
        # Tạo một client duy nhất và tái sử dụng connection pool
        # Tùy chọn decode_responses để phù hợp với cả worker và các service khác
        return redis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            decode_responses=decode_responses,
            health_check_interval=30, # Thêm health check để giữ kết nối ổn định
            password=self.redis_password
        )

# Tạo một instance duy nhất để import
redis_manager = RedisClient()

# Sử dụng:
# from app.lib.redis_client import redis_manager
# r = redis_manager.get_connection()
# r_decoded = redis_manager.get_connection(decode_responses=True)