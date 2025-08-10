# services/logging_config.py

import logging
from logging.handlers import TimedRotatingFileHandler
import os
import sys

# 📁 Tạo thư mục log nếu chưa có
log_dir = os.path.join(os.path.dirname(__file__), "..","..", "logs")
os.makedirs(log_dir, exist_ok=True)

# 📄 Đường dẫn các file log
all_log_file = os.path.join(log_dir, "workers_log.txt")
error_log_file = os.path.join(log_dir, "errors.txt") # ✅ ĐỊNH NGHĨA FILE LOG LỖI

# 🔧 Tạo logger
logger = logging.getLogger("worker_logger")
logger.setLevel(logging.INFO) # ✅ Set level tổng thể của logger là INFO (để bắt cả INFO và ERROR)

# Chỉ cấu hình handler một lần duy nhất
if not logger.handlers:
    # --- Định dạng chung cho tất cả các handler ---
    log_format = "%(asctime)s - %(levelname)s - [%(process)d] - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s"
    formatter = logging.Formatter(
        fmt=log_format,
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # --- Handler 1: Ghi TẤT CẢ log (từ INFO trở lên) ra file workers_log.txt ---
    all_log_handler = TimedRotatingFileHandler(
        filename=all_log_file,
        when="midnight",
        interval=1,
        backupCount=7,
        encoding="utf-8"
    )
    all_log_handler.suffix = "%Y-%m-%d"
    all_log_handler.setFormatter(formatter)
    all_log_handler.setLevel(logging.INFO) # Handler này xử lý từ INFO trở lên
    logger.addHandler(all_log_handler)

    # --- Handler 2: Ghi TẤT CẢ log (từ INFO trở lên) ra Console (stdout) ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO) # Handler này xử lý từ INFO trở lên
    logger.addHandler(console_handler)

    # --- ✅ Handler 3: Ghi CHỈ CÁC LỖI (từ ERROR trở lên) ra file errors.txt ---
    error_handler = logging.FileHandler(error_log_file, mode='a', encoding='utf-8')
    error_handler.setFormatter(formatter)
    error_handler.setLevel(logging.ERROR) # ⭐️ Đây là điểm mấu chốt: chỉ bắt ERROR và CRITICAL
    logger.addHandler(error_handler)


def get_logger():
    return logger