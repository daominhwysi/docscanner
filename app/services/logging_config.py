# services/logging_service.py

import logging
from logging.handlers import TimedRotatingFileHandler
import os

# 📁 Tạo thư mục log nếu chưa có
log_dir = os.path.join(os.path.dirname(__file__), "..","..", "logs")
os.makedirs(log_dir, exist_ok=True)

# 📄 Đường dẫn file log gốc (sẽ được rotate thành workers_log.txt.YYYY-MM-DD)
log_file = os.path.join(log_dir, "workers_log.txt")

# 🔧 Tạo một logger riêng
logger = logging.getLogger("worker_logger")
logger.setLevel(logging.INFO)

if not logger.handlers:
    # ✍️ Handler xoay vòng theo ngày, với suffix là ngày YYYY-MM-DD
    handler = TimedRotatingFileHandler(
        filename=log_file,
        when="midnight",
        interval=1,
        backupCount=7,       # giữ 7 ngày log cũ, bạn chỉnh tuỳ ý
        encoding="utf-8",
        utc=False            # nếu muốn theo giờ local, để False
    )
    # đặt định dạng suffix của file rotated
    handler.suffix = "%Y-%m-%d"

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)

    # ⚙️ Chỉ add file handler, không add StreamHandler
    logger.addHandler(handler)

def get_logger():
    return logger
