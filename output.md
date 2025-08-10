# Project Analysis Prompt

Analyze the following project structure and file contents.

## From: app

---
File: __pycache__/api.cpython-310.pyc
---
Error reading file: 'utf-8' codec can't decode byte 0x8e in position 10: invalid start byte

---
File: __pycache__/main.cpython-310.pyc
---
Error reading file: 'utf-8' codec can't decode byte 0x93 in position 10: invalid start byte

---
File: __pycache__/run_worker.cpython-310.pyc
---
Error reading file: 'utf-8' codec can't decode bytes in position 9-10: invalid continuation byte

---
File: __pycache__/worker_settings.cpython-310.pyc
---
Error reading file: 'utf-8' codec can't decode byte 0x8a in position 10: invalid start byte

---
File: db/__pycache__/client.cpython-310.pyc
---
Error reading file: 'utf-8' codec can't decode byte 0x91 in position 10: invalid start byte

---
File: db/__pycache__/models.cpython-310.pyc
---
Error reading file: 'utf-8' codec can't decode byte 0x91 in position 10: invalid start byte

---
File: db/client.py
---
```python
# db/client.py
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.db.models import Base

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# Create engine and sessionmaker
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

def init_db():
    Base.metadata.create_all(bind=engine)

# trong app/db/client.py
from contextlib import contextmanager


@contextmanager
def get_session():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
```

---
File: db/models.py
---
```python
# db/models.py
from sqlalchemy import (
    Column, String, Integer, DateTime, Enum as PgEnum,
    ForeignKey, Text
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime, timezone
from enum import Enum
import uuid

Base = declarative_base()

# Enums
class TaskType(str, Enum):
    parseDocumentPDF = "parseDocumentPDF"
    documentParsing = "documentParsing"

class TaskStatus(str, Enum):
    pending = "pending"
    done = "done"
    failed = "failed"
    cancelled = "cancelled"

# Models
class Task(Base):
    __tablename__ = "task"

    id = Column(String, primary_key=True, default=lambda: uuid.uuid4().hex)
    type = Column(PgEnum(TaskType), nullable=False)
    status = Column(PgEnum(TaskStatus), default=TaskStatus.pending, nullable=False)
    result = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    createdAt = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updatedAt = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    inference_logs = relationship("InferenceLog", back_populates="task", cascade="all, delete-orphan")

class InferenceLog(Base):
    __tablename__ = "inferencelog"

    id = Column(String, primary_key=True, default=lambda: uuid.uuid4().hex)
    imageUrls = Column(Text, nullable=True)
    rawOutput = Column(Text, nullable=False)
    objectKeys = Column(ARRAY(String), nullable=True)
    objectUrls = Column(ARRAY(String), nullable=True)
    num_input_token = Column(Integer, nullable=False)
    num_output_token = Column(Integer, nullable=False)
    page_order = Column(Integer, nullable=True)
    createdAt = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    requestId = Column(String, ForeignKey("task.id"), nullable=True)
    task = relationship("Task", back_populates="inference_logs")
```

---
File: lib/__pycache__/redis_client.cpython-310.pyc
---
Error reading file: 'utf-8' codec can't decode byte 0x85 in position 9: invalid start byte

---
File: lib/__pycache__/worker_core.cpython-310.pyc
---
Error reading file: 'utf-8' codec can't decode byte 0xb7 in position 8: invalid start byte

---
File: lib/redis_client.py
---
```python
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
```

---
File: lib/worker_core.py
---
```python
import asyncio
import json
from typing import Callable, Awaitable, Dict, Optional
from redis.asyncio import Redis
import os
from app.lib.redis_client import redis_manager

class SimpleRedisWorker:
    def __init__(self, queue_name: str = "tasks"):
        self.redis = redis_manager.get_connection(decode_responses=True)
        self.queue_name = queue_name
        self.tasks: Dict[str, Callable[..., Awaitable]] = {}
        self.semaphores: Dict[str, asyncio.Semaphore] = {}
        self.retries: Dict[str, int] = {}
        self.running = False

    def task(
        self,
        name: Optional[str] = None,
        max_concurrency: Optional[int] = None,
        max_retries: int = 0
    ):
        """Decorator để đăng ký task với optional concurrency limit và retry."""
        def decorator(func: Callable[..., Awaitable]):
            task_name = name or func.__name__
            self.tasks[task_name] = func
            if max_concurrency:
                self.semaphores[task_name] = asyncio.Semaphore(max_concurrency)
            if max_retries > 0:
                self.retries[task_name] = max_retries
            return func
        return decorator

    async def enqueue(self, task_name: str, *args, **kwargs):
        data = {"task": task_name, "args": args, "kwargs": kwargs}
        await self.redis.rpush(self.queue_name, json.dumps(data))

    async def run_worker(self):
        print(f"[Worker] Listening on queue '{self.queue_name}'...")
        self.running = True
        while self.running:
            item = await self.redis.blpop(self.queue_name, timeout=1)
            if not item:
                continue

            _, raw = item
            job = json.loads(raw)
            task_name = job.get("task")
            func = self.tasks.get(task_name)

            if not func:
                print(f"[Worker] Unknown task: {task_name}")
                continue

            args = job.get("args", [])
            kwargs = job.get("kwargs", {})
            sem = self.semaphores.get(task_name)
            max_retries = self.retries.get(task_name, 0)

            if sem:
                asyncio.create_task(
                    self._execute_with_limit_and_retry(sem, func, args, kwargs, max_retries)
                )
            else:
                asyncio.create_task(
                    self._execute_with_retry(func, args, kwargs, max_retries)
                )

    async def _execute_with_limit_and_retry(
        self,
        sem: asyncio.Semaphore,
        func: Callable[..., Awaitable],
        args: list,
        kwargs: dict,
        max_retries: int
    ):
        async with sem:
            await self._execute_with_retry(func, args, kwargs, max_retries)

    async def _execute_with_retry(
        self,
        func: Callable[..., Awaitable],
        args: list,
        kwargs: dict,
        max_retries: int
    ):
        attempt = 0
        while True:
            try:
                await func(*args, **kwargs)
                return
            except Exception as e:
                attempt += 1
                if attempt <= max_retries:
                    print(f"[Worker] Task '{func.__name__}' failed (attempt {attempt}/{max_retries}), retrying...")
                    await asyncio.sleep(1)  # backoff
                    continue
                else:
                    print(f"[Worker] Task '{func.__name__}' failed after {max_retries} retries: {e}")
                    return

    async def stop(self):
        self.running = False
```

---
File: main.py
---
```python
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header
from fastapi.responses import FileResponse, HTMLResponse
from pathlib import Path
from contextlib import asynccontextmanager

from blake3 import blake3
import aiofiles
import os
import httpx
import asyncio

from pydantic import BaseModel
from app.worker.instance import worker

from app.db.client import init_db, get_session
from app.services.create_task import create_task, get_task_by_id
from dotenv import load_dotenv
from app.db.models import Task, TaskType
from app.postprocessing.convert2html import convert_text2html
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import Optional
import aiohttp
import os
from app.utils.agent import init_api_key_zset# --- Cấu hình thư mục ---
PRIVATE_DIR = Path("media").resolve()
PRIVATE_DIR.mkdir(exist_ok=True)
keys = os.environ.get("GEMINI_API_KEY", "").split(",")

# --- Init DB khi start server ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    await init_api_key_zset(keys)
    yield

app = FastAPI(lifespan=lifespan)
# --- Middleware xác thực (giả lập) ---
def verify_access(x_token: str = Header(None)):
    if x_token != os.environ.get("X_FILE_TOKEN"):
        raise HTTPException(status_code=401, detail="Unauthorized")


# --- Serve file PRIVATE (PDF, ảnh gốc) ---
@app.get("/media/{file_path:path}")
def serve_private_file(file_path: str, _: str = Depends(verify_access)):
    full_path = (PRIVATE_DIR / file_path).resolve()

    if not str(full_path).startswith(str(PRIVATE_DIR)):
        raise HTTPException(status_code=403, detail="Access denied")

    if not full_path.exists() or not full_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(full_path, filename=full_path.name)



async def save_file(file_bytes: bytes, file_type: str) -> Path:
    # Tính hash
    file_hash = blake3(file_bytes).hexdigest()

    # Tạo thư mục con
    subdir = PRIVATE_DIR / file_hash[:2] / file_hash[2:4]
    subdir.mkdir(parents=True, exist_ok=True)

    filename = f"{file_hash}.{file_type}"
    filepath = subdir / filename

    # Chỉ ghi file nếu nó chưa tồn tại
    if not filepath.exists():
        temp_path = subdir / f".{filename}.tmp"
        # Ghi vào file tạm
        async with aiofiles.open(temp_path, "wb") as f:
            await f.write(file_bytes)
        # Rename file tạm thành file thật (atomic)
        try:
            temp_path.rename(filepath)
        except Exception as e:
            # Có thể file đã được tạo bởi một process khác trong lúc ghi, không sao cả
            print(f"Could not rename temp file, possibly due to race condition: {e}")
            if temp_path.exists():
                temp_path.unlink() # Xóa file tạm đi

    # Luôn tính toán và trả về đường dẫn tương đối
    relative_path = filepath.relative_to(PRIVATE_DIR)
    return relative_path

@app.get("/logs")
def read_latest_worker_log():
    # 📁 Trỏ tới thư mục chứa log
    log_dir = (Path(__file__).resolve().parent.parent / "logs").resolve()

    if not log_dir.exists() or not log_dir.is_dir():
        raise HTTPException(status_code=404, detail="Log directory not found")

    # 🔍 Tìm tất cả các file bắt đầu bằng "workers_log"
    log_files = list(log_dir.glob("workers_log*.txt*"))

    if not log_files:
        raise HTTPException(status_code=404, detail="No log files found")

    # 🕓 Chọn file mới nhất dựa vào thời gian sửa đổi
    latest_log = max(log_files, key=lambda f: f.stat().st_mtime)

    return FileResponse(
        path=latest_log,
        media_type="text/plain",
        filename=latest_log.name,
        headers={"Content-Disposition": f"inline; filename={latest_log.name}"}
    )
async def get_file_bytes(url):
    headers = {"x-token": os.environ.get("X_FILE_TOKEN")}
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers=headers)
        return resp.content
# --- API Upload PDF ---


@app.post("/process-pdf")
async def handle_pdf(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None)  # Dùng Form để nhận URL từ body
):
    file_bytes = None
    filename = None

    if file:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Chỉ chấp nhận file PDF.")
        file_bytes = await file.read()
        filename = file.filename
        await file.close()
    elif url:
        if not url.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="URL phải trỏ đến một file PDF.")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        raise HTTPException(status_code=400, detail="Không thể tải file từ URL.")
                    file_bytes = await resp.read()
                    filename = os.path.basename(url)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Lỗi khi tải file: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="Cần cung cấp file hoặc URL.")

    # Lưu file
    relative_file_path_str = await save_file(file_bytes=file_bytes, file_type='pdf')
    file_url = f"http://localhost:8000/media/{relative_file_path_str}"

    # Tạo task
    task_id = None
    with get_session() as session:
        task = create_task(task_type=TaskType.parseDocumentPDF, session=session)
        task_id = task.id

    await worker.enqueue("process_pdf", task_id, file_url)

    return {"task_id": task_id}


class Text2Slurp(BaseModel):
    text: str

@app.post("/document-parsing")
async def documentParsing(body: Text2Slurp):
    task_id=None
    with get_session() as session:
        task = create_task(task_type=TaskType.documentParsing, session=session)
        task_id = task.id
    await worker.enqueue("documentParsing", task_id, body.text)
    return {"task_id": task_id}


# Route: GET one task theo id
@app.get("/tasks/{task_id}")
def get_task(task_id: str):
    with get_session() as session:
        task = get_task_by_id(task_id=task_id, session=session)
        if task:
                return {
                    "id": str(task.id),
                    "type": task.type.value,
                    "status": task.status.value,
                    "result": task.result,
                    "createdAt": task.createdAt.isoformat(),
                    "updatedAt": task.updatedAt.isoformat(),
                }
        raise HTTPException(status_code=404, detail="Task not found")

class Text2Html(BaseModel):
    text: str

@app.post("/text2html/")
async def text2html(body: Text2Html):
    
    return HTMLResponse(content=convert_text2html(body.text) , media_type="text/html")

# # --- Serve file PUBLIC (figure) ---
# @app.get("/media-public/{file_path:path}")
# def serve_public_file(file_path: str):
#     full_path = (PUBLIC_DIR / file_path).resolve()

#     if not str(full_path).startswith(str(PUBLIC_DIR)):
#         raise HTTPException(status_code=403, detail="Access denied")

#     if not full_path.exists() or not full_path.is_file():
#         raise HTTPException(status_code=404, detail="File not found")

#     return FileResponse(full_path, filename=full_path.name)
```

---
File: ml_models/__init__.py
---
```python

```

---
File: ml_models/__pycache__/__init__.cpython-310.pyc
---
Error reading file: 'utf-8' codec can't decode byte 0x91 in position 10: invalid start byte

---
File: ml_models/__pycache__/image_classifier.cpython-310.pyc
---
Error reading file: 'utf-8' codec can't decode byte 0x91 in position 10: invalid start byte

---
File: ml_models/__pycache__/rfdetr.cpython-310.pyc
---
Error reading file: 'utf-8' codec can't decode byte 0x91 in position 10: invalid start byte

---
File: ml_models/image_classifier.py
---
```python
import onnxruntime
import numpy as np
import cv2
import os

# --- Config ---
IMAGE_SIZE = 512
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ONNX_MODEL_PATH = os.path.join(BASE_DIR, 'weights', 'ghostnet_classifier.onnx')
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
THRESHOLD_POSITIVE = 0.65


# --- Image Classifier Class ---
class ImageClassifier:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found at {model_path}")
        self.session = onnxruntime.InferenceSession(model_path)
        print(f"Model loaded: {model_path}")

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def preprocess(self, np_img):
        img = cv2.resize(np_img, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = img.astype(np.float32)
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        return np.expand_dims(img, axis=0)  # (1, 3, H, W)

    def predict(self, np_img: np.ndarray):
        try:
            input_tensor = self.preprocess(np_img)
            ort_inputs = {self.session.get_inputs()[0].name: input_tensor}
            ort_outputs = self.session.run(None, ort_inputs)
            logits = ort_outputs[0]
            probs = self.softmax(logits)
            confidence = probs[0][1]
            predicted_class_index = 1 if confidence > THRESHOLD_POSITIVE else 0
            return predicted_class_index, logits, confidence
        except Exception as e:
            print(f"Lỗi khi xử lý ảnh numpy: {e}")
            return None, None, None

classifier = ImageClassifier(ONNX_MODEL_PATH)

# # --- Test ---
# if __name__ == "__main__":
#     image_path = "your_image.png"  # ← Thay bằng ảnh của bạn
#     if not os.path.exists(image_path):
#         print(f"Ảnh không tồn tại: {image_path}")
#         exit(1)

#     np_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
#     classifier = ImageClassifier(ONNX_MODEL_PATH)

#     class_to_idx = {0: "negative", 1: "positive"}
#     pred_idx, logits, confidence = classifier.predict(np_image)

#     if pred_idx is not None:
#         print(f"Prediction: {class_to_idx[pred_idx]} (Confidence: {confidence:.2%})")
#     else:
#         print("Không thể dự đoán.")
```

---
File: ml_models/rfdetr.py
---
```python
# This file contains code licensed under the Apache License, Version 2.0.
# See NOTICE for more details.

import io
import requests
import onnxruntime as ort
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def box_cxcywh_to_xyxy_numpy(x):
    x_c, y_c, w, h = np.split(x, 4, axis=-1)
    b = np.concatenate([
        x_c - 0.5 * np.clip(w, a_min=0.0, a_max=None),
        y_c - 0.5 * np.clip(h, a_min=0.0, a_max=None),
        x_c + 0.5 * np.clip(w, a_min=0.0, a_max=None),
        y_c + 0.5 * np.clip(h, a_min=0.0, a_max=None)
    ], axis=-1)
    return b

class RTDETR_ONNX:
    MEANS = [0.485, 0.456, 0.406]
    STDS = [0.229, 0.224, 0.225]

    def __init__(self, onnx_model_path):

        # Load the ONNX model and initialize the ONNX Runtime session
        self.ort_session = ort.InferenceSession(onnx_model_path)
        print(f'Model RFDETR loaded: {onnx_model_path}')
        # Get input shape
        input_info = self.ort_session.get_inputs()[0]
        self.input_height, self.input_width = input_info.shape[2:]


    def _preprocess_image(self, image):
        """Preprocess the input image for inference."""

        # Resize the image to the model's input size
        image = image.resize((self.input_width, self.input_height))

        # Convert image to numpy array and normalize pixel values
        image = np.array(image).astype(np.float32) / 255.0

        # Normalize
        image = ((image - self.MEANS) / self.STDS).astype(np.float32)

        # Change dimensions from HWC to CHW
        image = np.transpose(image, (2, 0, 1))

        # Add batch dimension
        image = np.expand_dims(image, axis=0)

        return image

    def _post_process(self, outputs, origin_height, origin_width, confidence_threshold, max_number_boxes):
        """Post-process the model's output to extract bounding boxes and class information."""
        # Get the bounding box and class scores
        pred_boxes, pred_logits = outputs

        # Apply sigmoid activation
        prob = sigmoid(pred_logits)

        # Get the top-k values and indices
        flat_prob = prob[0].flatten()
        topk_indexes = np.argsort(flat_prob)[-max_number_boxes:][::-1]
        topk_values = np.take_along_axis(flat_prob, topk_indexes, axis=0)
        scores = topk_values
        topk_boxes = topk_indexes // pred_logits.shape[2]
        labels = topk_indexes % pred_logits.shape[2]

        # Gather boxes corresponding to top-k indices
        boxes = box_cxcywh_to_xyxy_numpy(pred_boxes[0])
        boxes = np.take_along_axis(boxes, np.expand_dims(topk_boxes, axis=-1).repeat(4, axis=-1), axis=0)

        # Rescale box locations
        target_sizes = np.array([[origin_height, origin_width]])
        img_h, img_w = target_sizes[:, 0], target_sizes[:, 1]
        scale_fct = np.stack([img_w, img_h, img_w, img_h], axis=1)
        boxes = boxes * scale_fct[0, :]

        # Filter detections based on the confidence threshold
        high_confidence_indices = np.argmin(scores > confidence_threshold)
        scores = scores[:high_confidence_indices]
        labels = labels[:high_confidence_indices]
        boxes = boxes[:high_confidence_indices]

        return scores, labels, boxes

    def run_inference(self, image, confidence_threshold=0.5, max_number_boxes=100):
        """Run the model inference and return the raw outputs."""

        # Load the image
        image = image.convert('RGB')
        origin_width, origin_height = image.size

        # Preprocess the image
        input_image = self._preprocess_image(image)

        # Get input name from the model
        input_name = self.ort_session.get_inputs()[0].name

        # Run the model
        outputs = self.ort_session.run(None, {input_name: input_image})

        # Post-process
        return self._post_process(outputs, origin_height, origin_width, confidence_threshold, max_number_boxes)


    def save_detections(self, image, boxes, labels, save_image_path):
        """Draw bounding boxes and class labels on the original image."""
        # Load the original image
        image = image.convert('RGB')

        draw = ImageDraw.Draw(image)

        # Loop over the boxes
        for i, box in enumerate(boxes.astype(int)):

            # Draw the rectangle (box) on the image
            draw.rectangle(box.tolist(), outline="green", width=4)

            # Using default font
            font = ImageFont.load_default()

            # Position the text inside the rectangle
            text_x = box[0] + 10  # Left margin for text
            text_y = box[1] + 10  # Top margin for text
            draw.text((text_x, text_y), str(labels[i]), fill="red", font=font)

        # Save the image with the rectangle and text
        image.save(save_image_path)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'weights', 'rfdetr-figure.onnx')
rtdetr_model = RTDETR_ONNX(MODEL_PATH)
```

---
File: postprocessing/__pycache__/convert2html.cpython-310.pyc
---
Error reading file: 'utf-8' codec can't decode byte 0x91 in position 10: invalid start byte

---
File: postprocessing/__pycache__/raw_response.cpython-310.pyc
---
Error reading file: 'utf-8' codec can't decode byte 0x91 in position 10: invalid start byte

---
File: postprocessing/__pycache__/replaceimgfig.cpython-310.pyc
---
Error reading file: 'utf-8' codec can't decode byte 0xe2 in position 8: invalid continuation byte

---
File: postprocessing/__pycache__/slurp2json.cpython-310.pyc
---
Error reading file: 'utf-8' codec can't decode byte 0x84 in position 8: invalid start byte

---
File: postprocessing/convert2html.py
---
```python
import re
from latex2mathml.converter import convert


def latex_to_inline_mathml(text: str) -> str:
    pattern = re.compile(r"""
        \\\[(.+?)\\\]      |   # group 2: \[...\]
        \\\((.+?)\\\)      |   # group 3: \(...\)
    """, re.VERBOSE | re.DOTALL)

    def repl(m):
        groups = [g for g in m.groups() if g is not None]
        if not groups:
            # Không có gì match, trả nguyên đoạn
            return m.group(0)
        
        latex = groups[0].strip()
        try:
            mathml = convert(latex)
            return f'{mathml}'
        except Exception:
            return m.group(0)


    return pattern.sub(repl, text)

def convert_text_to_html(text):
    lines = text.strip().split('\n')
    html_lines = []

    for line in lines:
        html_lines.append(f'<p>{line}</p>')

    return ''.join(html_lines)

def convert_text2html(input_data: str):
    text_mathml = latex_to_inline_mathml(input_data)
    html = convert_text_to_html(text_mathml)
    return html
```

---
File: postprocessing/raw_response.py
---
```python
import re
from typing import Any, Dict, Tuple, Union, Optional


def find_last_tag_block(text: str, tag_name: str) -> Optional[str]:
    """
    Finds the content of the last block between <tag_name>...</tag_name> tags.
    Case-insensitive, supports tags with attributes.
    Returns the content between the last matching tags, or None if not found.
    """
    # Escape tag name for regex
    escaped = re.escape(tag_name)
    open_pattern = re.compile(rf"<\s*{escaped}\b[^>]*>", re.IGNORECASE)
    close_pattern = re.compile(rf"</\s*{escaped}\s*>", re.IGNORECASE)

    # Find all closing tags
    closes = list(close_pattern.finditer(text))
    if not closes:
        return None

    last_close = closes[-1]
    close_start = last_close.start()

    # Find opening tags before last close
    opens = []
    for m in open_pattern.finditer(text):
        if m.start() < close_start:
            opens.append(m)
        else:
            break

    if not opens:
        return None

    last_open = opens[-1]
    content_start = last_open.end()
    return text[content_start:close_start].strip()


def extract_and_remove_thinking_block(text: str) -> Tuple[str, str]:
    """
    Extracts the content of the outermost <thinking>...</thinking> block.
    Returns a tuple: (text_without_block, inner_content)
    """
    # Find all closing thinking tags
    close_pattern = re.compile(r"</thinking\s*>", re.IGNORECASE)
    closes = list(close_pattern.finditer(text))
    if not closes:
        return text, ""

    last_close = closes[-1]
    close_start = last_close.start()
    close_end = last_close.end()

    # Find first opening thinking tag
    open_pattern = re.compile(r"<thinking[^>]*>", re.IGNORECASE)
    open_match = open_pattern.search(text)
    if not open_match or open_match.start() >= close_start:
        return text, ""

    open_start = open_match.start()
    open_end = open_match.end()

    inner = text[open_end:close_start]
    outside = text[:open_start] + text[close_end:]
    return outside, inner


class ExtractedResponse:
    def __init__(self, thinking_block: str, document: Optional[str]):
        self.thinking_block = thinking_block
        self.document = document


def extract_response(text: str) -> ExtractedResponse:
    """
    Extracts the thinking block and the final document content from raw text.
    Returns an ExtractedResponse with thinking_block and document.
    """
    cleaned, thinking = extract_and_remove_thinking_block(text)
    block = find_last_tag_block(cleaned, "assessmentmarkuplanguage")
    doc: Optional[str] = None
    doc = block
    return ExtractedResponse(thinking_block=thinking, document=doc)


def replace_image_tags(
    content: str,
    image_dict: dict[str, str]
) -> Tuple[str, str]:
    """
    Replace <graphic tag='IMx' label='...'> with <img src='...'
    alt='...'/> using URLs from image_dict and preserving label from original tag.
    """
    if not isinstance(image_dict, dict):
        return content, 404

    pattern = re.compile(
        r"<graphic\s+tag=['\"]?(IM[0-9O]+)['\"]?"         # group 1: tag
        r"(?:\s+label=['\"](.*?)['\"])?\s*/?>",           # group 2: optional label
        re.IGNORECASE
    )

    used = set()
    missing = False
    extra = False

    def normalize_key(raw: str) -> str:
        return raw.upper().replace('O', '0')

    def repl(match: re.Match) -> str:
        nonlocal missing, extra
        raw_key = match.group(1)
        label = match.group(2)
        key = normalize_key(raw_key)

        url = image_dict.get(key)
        if not url:
            extra = True
            return ''  # hoặc match.group(0) để giữ nguyên nếu muốn

        used.add(key)
        alt = label or key
        return f'<img src="{url}" alt="{alt}"/>'

    try:
        new_content = pattern.sub(repl, content)
    except Exception:
        return content, 404

    # Kiểm tra khóa không dùng tới
    dict_keys = {normalize_key(k) for k in image_dict.keys()}
    if dict_keys - used:
        missing = True

    status = 200 if not (missing or extra) else 404
    return new_content, status


if __name__ == "__main__":
    # Example usage
    sample = "<thinking>Compute something</thinking>..."
    res = extract_response(sample)
    print(res.thinking_block, res.document)
```

---
File: postprocessing/replaceimgfig.py
---
```python
import re
from typing import List, Dict, Tuple, Any

def replace_img_to_fig(html: str) -> Tuple[str, List[Dict[str, str]]]:
    image_srcs = []
    index = 0

    def replacer(match):
        nonlocal index
        full_tag = match.group(0)  # toàn bộ thẻ <img ...>
        src = match.group(1)
        attrs = match.group(2) or ""

        img_id = f"fig{index}"
        image_srcs.append({"id": img_id, "src": src})
        index += 1

        # Thay thế bằng <figure id="figX" ... />
        return f'<figure id="{img_id}" {attrs.strip()} />'

    pattern = r'<img\s+[^>]*src=["\']([^"\']+)["\']([^>]*)>'
    text = re.sub(pattern, replacer, html, flags=re.IGNORECASE)

    return text, image_srcs

def replace_fig2img_immutable(json_obj: Any, figures_data: List[Dict[str, str]]) -> Any:
    id_to_src_map = {figure['id']: figure['src'] for figure in figures_data}
    FIGURE_TAG_RE = re.compile(
        r'<figure\b[^>]*\bid\s*=\s*["\']([^"\']+)["\'][^>]*>',
        re.IGNORECASE
    )

    def replace_figures_in_text(text: str) -> str:
        def replacer(match):
            fig_id = match.group(1)
            src = id_to_src_map.get(fig_id)
            if src:
                return f'<img src="{src}">'
            return match.group(0)
        return FIGURE_TAG_RE.sub(replacer, text)

    def _walk_and_rebuild(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: _walk_and_rebuild(v) for k, v in node.items()}
        elif isinstance(node, list):
            return [_walk_and_rebuild(item) for item in node]
        elif isinstance(node, str):
            return replace_figures_in_text(node)
        else:
            return node

    return _walk_and_rebuild(json_obj)

if __name__ == "__main__":
    figures = [{'id': 'fig0', 'src': 'https://pub-81feb243d1ab4aae8e02911485784de0.r2.dev/a25fdf9b.jpg'}, {'id': 'fig1', 'src': 'https://pub-81feb243d1ab4aae8e02911485784de0.r2.dev/29cccdd3.jpg'}, {'id': 'fig2', 'src': 'https://pub-81feb243d1ab4aae8e02911485784de0.r2.dev/e88ed63b.jpg'}, {'id': 'fig3', 'src': 'https://pub-81feb243d1ab4aae8e02911485784de0.r2.dev/7ae81bb9.jpg'}, {'id': 'fig4', 'src': 'https://pub-81feb243d1ab4aae8e02911485784de0.r2.dev/a1dcd944.jpg'}, {'id': 'fig5', 'src': 'https://pub-81feb243d1ab4aae8e02911485784de0.r2.dev/31d6ba4f.jpg'}]
    with open("/teamspace/studios/this_studio/tests/de15.txt", "r", encoding="utf-8") as f:
        text = f.read()
```

---
File: postprocessing/slurp2json.py
---
```python
import re
import json
from typing import List, Dict, Any, Union, Optional, Literal, TypedDict

# TypeScript interface: Block
# Được biểu diễn bằng TypedDict để an toàn kiểu
class Block(TypedDict):
    type: str
    content: Union[str, Dict[str, Any]]

# TypeScript interface: State
class State(TypedDict):
    currentBlockType: Optional[str]
    currentBlockObject: Optional[Union[str, Dict[str, Any]]]
    inMultiLine: bool
    multiLineTarget: Optional[Literal['block', 'property']]
    propertyStack: List[str]

class SLURPParser:
    """
    Phân tích cú pháp văn bản định dạng SLURP thành một cấu trúc các khối (blocks).
    SLURP là một định dạng đơn giản, thụt đầu dòng để biểu diễn các khối dữ liệu lồng nhau.
    """
    _BLOCK_REGEX = re.compile(r"^(\w+):\s*\|?(.*)$")
    _PROP_REGEX = re.compile(r"^(>+)\s*([\w-]+):\s*\|?(.*)$")

    def __init__(self):
        self.result: List[Block] = []
        self.current_state: State = self._get_initial_state()

    def _get_initial_state(self) -> State:
        """Trả về một từ điển trạng thái ban đầu."""
        return {
            "currentBlockType": None,
            "currentBlockObject": None,
            "inMultiLine": False,
            "multiLineTarget": None,
            "propertyStack": [],
        }

    def parse(self, text_content: str) -> List[Block]:
        """
        Phân tích cú pháp một chuỗi đầu vào SLURP và trả về một danh sách các khối.
        """
        self.result = []
        self.current_state = self._get_initial_state()

        lines = text_content.replace('\r\n', '\n').split('\n')

        for line in lines:
            if not line.strip() and not self.current_state["inMultiLine"]:
                continue
            self._process_line(line)

        self._finalize_current_block()
        return self.result

    # ========================================================================
    # HÀM ĐÃ ĐƯỢỢC SỬA LỖI BÊN DƯỚI
    # ========================================================================
    def _process_line(self, line: str) -> None:
        """Xử lý một dòng đơn từ đầu vào SLURP."""
        block_match = self._BLOCK_REGEX.match(line)
        prop_match = self._PROP_REGEX.match(line)

        # Phát hiện khối mới
        if not line.startswith('>') and not line.startswith(' ') and block_match:
            self._finalize_current_block()

            block_type, value = block_match.groups()
            self.current_state["currentBlockType"] = block_type
            trimmed_value = value.strip()

            if trimmed_value or '|' in line:
                self.current_state["currentBlockObject"] = trimmed_value
                if '|' in line:
                    self.current_state["inMultiLine"] = True
                    self.current_state["multiLineTarget"] = 'block'
            else:
                self.current_state["currentBlockObject"] = {}
                self.current_state["inMultiLine"] = False

        # Phát hiện thuộc tính
        elif prop_match:
            self.current_state["inMultiLine"] = False

            gt, key, value = prop_match.groups()
            level = len(gt)

            # Cắt ngắn ngăn xếp thuộc tính về mức độ thụt đầu dòng hiện tại
            self.current_state["propertyStack"] = self.current_state["propertyStack"][:level - 1]

            parent_obj = self.current_state["currentBlockObject"]
            if not isinstance(parent_obj, dict):
                # Tự động chuyển đổi chuỗi thành đối tượng nếu cần
                parent_obj = {}
                self.current_state["currentBlockObject"] = parent_obj

            # *** PHẦN SỬA LỖI QUAN TRỌNG ***
            # Đoạn code này đảm bảo rằng chúng ta đang đi vào một từ điển (dict).
            # Nếu đường dẫn tồn tại nhưng chứa một giá trị không phải dict (ví dụ: một chuỗi rỗng),
            # nó sẽ được ghi đè bằng một dict rỗng.
            current_level_ref = parent_obj
            for path_key in self.current_state["propertyStack"]:
                # Kiểm tra xem khóa có tồn tại không và giá trị của nó có phải là dict không
                if not isinstance(current_level_ref.get(path_key), dict):
                    # Nếu không, tạo/ghi đè nó bằng một dict rỗng
                    current_level_ref[path_key] = {}
                # Đi sâu vào cấp độ tiếp theo
                current_level_ref = current_level_ref[path_key]
            # *** KẾT THÚC PHẦN SỬA LỖI ***

            # Bây giờ `current_level_ref` chắc chắn là một dict
            current_level_ref[key] = value.strip()
            self.current_state["propertyStack"].append(key)

            if '|' in line:
                self.current_state["inMultiLine"] = True
                self.current_state["multiLineTarget"] = 'property'

        # Xử lý nội dung nhiều dòng
        elif self.current_state["inMultiLine"]:
            content = line
            if self.current_state["multiLineTarget"] == 'block':
                if isinstance(self.current_state["currentBlockObject"], str):
                    self.current_state["currentBlockObject"] += '\n' + content
            elif self.current_state["multiLineTarget"] == 'property':
                parent = self.current_state["currentBlockObject"]
                if isinstance(parent, dict):
                    stack = self.current_state["propertyStack"]
                    if not stack: return # Bỏ qua nếu không có thuộc tính nào trong ngăn xếp

                    # Đi đến cha của thuộc tính cuối cùng
                    for i in range(len(stack) - 1):
                        # Giả sử đường dẫn đã hợp lệ vì nó được tạo ở trên
                        parent = parent[stack[i]]

                    final_key = stack[-1]
                    if final_key in parent and isinstance(parent[final_key], str):
                        parent[final_key] += '\n' + content
    # ========================================================================
    # KẾT THÚC HÀM ĐÃ ĐƯỢC SỬA LỖI
    # ========================================================================

    def _finalize_current_block(self) -> None:
        """Lưu khối hiện đang được xử lý vào kết quả."""
        current_block_type = self.current_state["currentBlockType"]
        current_block_object = self.current_state["currentBlockObject"]

        if current_block_type and current_block_object is not None:
            final_content = current_block_object

            if isinstance(final_content, str):
                final_content = final_content.strip()

            self.result.append({
                "type": current_block_type,
                "content": final_content
            })

        self.current_state = self._get_initial_state()


def format_output(parsed_array: List[Block]) -> str:
    """
    Chuyển đổi danh sách các khối đã phân tích cú pháp thành định dạng CSV phẳng tùy chỉnh.
    Định dạng: blockType[index].path.to.property,"value"
    """
    output_lines: List[str] = []
    counters: Dict[str, int] = {}

    def flatten(obj: Dict[str, Any], path: str):
        for key, value in obj.items():
            new_path = f"{path}.{key}" if path else key
            if value and isinstance(value, dict):
                flatten(value, new_path)
            else:
                # json.dumps trên một chuỗi sẽ thêm dấu ngoặc kép và thoát các ký tự đặc biệt
                formatted_value = json.dumps(str(value))
                output_lines.append(f"{new_path},{formatted_value}")

    for block in parsed_array:
        block_type = block['type']
        block_content = block['content']

        counters[block_type] = counters.get(block_type, 0) + 1
        index = counters[block_type]
        base_path = f"{block_type}[{index}]"

        if block_content and isinstance(block_content, dict):
            flatten(block_content, base_path)
        else:
            formatted_value = json.dumps(str(block_content))
            output_lines.append(f"{base_path},{formatted_value}")

    return '\n'.join(output_lines)


def split_csv_lines_safe(input_str: str) -> List[str]:
    """
    Tách các dòng CSV một cách an toàn, xử lý các dấu xuống dòng bên trong các chuỗi được trích dẫn.
    """
    lines: List[str] = []
    current_line = ""
    inside_quotes = False

    for i, char in enumerate(input_str):
        if char == '"':
            # Trong JS, input[-1] là undefined. Trong Python, đó là ký tự cuối cùng.
            # Vì vậy, chúng ta phải kiểm tra rõ ràng i > 0.
            is_escaped = i > 0 and input_str[i-1] == '\\'
            if not is_escaped:
                inside_quotes = not inside_quotes

        if char == '\n' and not inside_quotes:
            lines.append(current_line)
            current_line = ""
        else:
            current_line += char

    if current_line:
        lines.append(current_line)

    return lines

# Định nghĩa các loại để làm rõ, tương tự như DataObject và DataMap trong TypeScript
DataMap = Dict[str, List[Dict[str, Any]]]
_CSV_KEY_REGEX = re.compile(r"^(\w+)\[(\d+)\](?:\.(.+))?$")

def parse_csv_to_json(input_str: str) -> DataMap:
    """
    Phân tích cú pháp chuỗi định dạng CSV tùy chỉnh thành một cấu trúc đối tượng lồng nhau,
    đồng thời giữ lại thuộc tính __order để biết thứ tự xuất hiện ban đầu của mỗi khối.
    """
    lines = split_csv_lines_safe(input_str)
    data: DataMap = {}

    global_order = 0

    for line in lines:
        if not line:
            continue

        parts = line.split(',', 1)
        if len(parts) < 2:
            continue
        raw_key, raw_value = parts

        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value  # fallback nếu không phải JSON

        match = _CSV_KEY_REGEX.match(raw_key)
        if not match:
            continue

        obj_name, idx_str, path = match.groups()
        index = int(idx_str)

        if obj_name not in data:
            data[obj_name] = []

        # đảm bảo danh sách đủ dài
        if len(data[obj_name]) < index:
            data[obj_name].extend([None] * (index - len(data[obj_name])))

        list_index = index - 1

        # tạo object mới kèm __order nếu chưa có
        if data[obj_name][list_index] is None:
            data[obj_name][list_index] = {'__order': global_order}
            global_order += 1

        target = data[obj_name][list_index]

        if path:
            keys = path.split('.')
            # tạo nested dict nếu cần
            for key_part in keys[:-1]:
                target = target.setdefault(key_part, {})
            target[keys[-1]] = value
        else:
            target['value'] = value

    # loại bỏ các slot None
    for key in data:
        data[key] = [entry for entry in data[key] if entry is not None]

    return data



def slurp_to_json(slurp: str) -> Dict:
    """
    Chức năng điều phối chính chuyển đổi một chuỗi SLURP thành một chuỗi JSON.
    """
    parser = SLURPParser()
    parsed_blocks = parser.parse(slurp)
    csv_str = format_output(parsed_blocks)
    json_obj = parse_csv_to_json(csv_str)
    return json_obj

# ---- Ví dụ sử dụng ----
# if __name__ == "__main__":
#     BASE_PATH = "/teamspace/studios/this_studio/tests/"
#     with open(f"{BASE_PATH}raw.txt", "r", encoding="utf-8") as f:
#         text = f.read()

#     # Chuyển nội dung sang JSON (dict hoặc list)
#     parsed = slurp_to_json(text)

#     # Ghi ra file JSON đúng cách
#     with open(f"{BASE_PATH}output.json", "w", encoding="utf-8") as f:
#         json.dump(parsed, f, ensure_ascii=False, indent=4)


import re

def autofix_missing_pipes(text: str) -> str:
    lines = text.replace('\r\n', '\n').split('\n')
    fixed_lines = []

    block_regex = re.compile(r"^([^:]+):\s*(?!\|)(.*)$")  # match block chưa có |
    prop_regex = re.compile(r"^(>+)\s*([^:]+):\s*(?!\|)(.*)$")
    new_block_or_prop = re.compile(r"^([^:]+:|>+\s*[^:]+:)")

    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Nếu dòng này đã có | thì giữ nguyên, không xử lý
        if re.match(r"^[^:]+:\s*\|", line):
            fixed_lines.append(line)
            i += 1
            continue

        next_line = lines[i + 1] if i + 1 < len(lines) else ""

        is_block = block_regex.match(line)
        is_prop = prop_regex.match(line)

        if (is_block or is_prop) and next_line.strip() != "":
            if not new_block_or_prop.match(next_line):
                before_colon, after_colon = line.split(':', 1)
                fixed_lines.append(f"{before_colon}: |")
                if after_colon.strip():
                    fixed_lines.append(f"  {after_colon.strip()}")
                i += 1
                while i < len(lines) and (lines[i].startswith(" ") or lines[i].strip() == "") and not new_block_or_prop.match(lines[i]):
                    fixed_lines.append(lines[i])
                    i += 1
                continue

        fixed_lines.append(line)
        i += 1

    return '\n'.join(fixed_lines)


if __name__ == "__main__":
    sample_text = """title: Đây là tiêu đề
noidung: Dòng đầu tiên
  Dòng thứ hai
author: Minh"""

    print("=== Input ===")
    print(sample_text)
    print("\n=== Output ===")
    print(autofix_missing_pipes(sample_text))
    print(slurp_to_json(autofix_missing_pipes(sample_text)))
```

---
File: prompt/__init__.py
---
```python
import re
import os
from google.genai import types
BASE_DIR = os.path.dirname(__file__)  
# Tương đương với readTextFile
def read_text_file(dir_path, file_path):
    with open(os.path.join(dir_path, file_path), 'r', encoding='utf-8') as file:
        return file.read()

def process_examples_xml(content: str):
    matches = re.findall(r"<example>([\s\S]*?)<\/example>", content)
    examples = []

    for match in matches:
        user_query_match = re.search(r"<user_query>([\s\S]*?)<\/user_query>", match)
        assistant_response_match = re.search(r"<assistant_response>([\s\S]*?)<\/assistant_response>", match)

        user_query = user_query_match.group(1).strip() if user_query_match else ''
        assistant_response = assistant_response_match.group(1).strip() if assistant_response_match else ''

        examples.append({
            'user_query': user_query,
            'assistant_response': assistant_response
        })

    # Mapping to contents
    contents = []
    for example in examples:
        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=example['user_query']),
                ]
            )
        )

        contents.append(
            types.Content(
                role="model",
                parts=[
                    types.Part.from_text(text=example['assistant_response']),
                ]
            )
        )


    return contents

# Initial Slurp
def get_initial_slurp_prompt():
    return read_text_file(BASE_DIR, './convert2slurp/initial_slurp.md')

def get_initial_slurp_examples():
    content = read_text_file(BASE_DIR, './convert2slurp/initial_example.xml')
    return process_examples_xml(content)

# Continuation Prompt
def get_slurp_continuation_prompt():
    return read_text_file(BASE_DIR, './convert2slurp/continuation_slurp.md')

def get_slurp_continuation_examples():
    initial_example = process_examples_xml(read_text_file(BASE_DIR, './convert2slurp/initial_example.xml'))
    cont_example = process_examples_xml(read_text_file(BASE_DIR, './convert2slurp/continuation_example.xml'))
    return initial_example + cont_example

# Extraction Prompts
def get_extraction_figure_prompt():
    return read_text_file(BASE_DIR, './extraction/figure.txt')

def get_extraction_non_figure_prompt():
    return read_text_file(BASE_DIR, './extraction/non_docfigure.txt')
```

---
File: prompt/__pycache__/__init__.cpython-310.pyc
---
Error reading file: 'utf-8' codec can't decode byte 0x91 in position 10: invalid start byte

---
File: prompt/convert2slurp/continuation_example.xml
---
```xml
<examples>
<example>
<user_query>
@mml_start@Câu 1. Mặt trận Việt Minh được thành lập vào thời gian nào?
A. 3/2/1930
B. 22/12/1944
C. 19/5/1941
D. 2/9/1945
Câu 2. Tổ chức ASEAN được thành lập năm nào?
A. 1965
B. 1967
C. 1970
D. 1975@mml_end@

@slurp_incomplete_start@qs:
>dnum: 1
>type: mcq
>qt: Mặt trận Việt Minh được thành lập vào thời gian nào?
>items:
>>a: 3/2/1930
>>b: 22/12/1944
>>c: 19/5/1941
>>d: 2/9/1945
qs:
>dnum: 2
>type: mcq@slurp_incomplete_end@
</user_query>


<assistant_response>
@slurp_resume_start@>qt: Tổ chức ASEAN được thành lập năm nào?
>items:
>>a: 1965
>>b: 1967
>>c: 1970
>>d: 1975@slurp_resume_end@
</assistant_response>
</example>


<example>


<user_query>
@mml_start@Đọc thông tin sau và trả lời câu hỏi 1 và 2:

Quang hợp là quá trình thực vật sử dụng ánh sáng mặt trời để tổng hợp các chất hữu cơ từ CO₂ và H₂O, giải phóng O₂.

Câu 1. Quá trình quang hợp xảy ra ở bộ phận nào của cây?
A. Rễ
B. Thân
C. Lá
D. Hoa

Câu 2. Sản phẩm chính của quá trình quang hợp là gì?
A. Nước
B. CO₂
C. Chất hữu cơ
D. Ánh sáng@mml_end@

@slurp_incomplete_start@stimulus:
>id: s-photosynthesis
>info: Quang hợp là quá trình thực vật sử dụng ánh sáng mặt trời để tổng hợp các chất hữu cơ từ CO₂ và H₂O, giải phóng O₂.
qs:
>dnum: 1
>type: mcq
>stimulus: s-photosynthesis
>qt: Quá trình quang hợp xảy ra ở bộ phận nào của cây?
>items:
>>a: Rễ
>>b: Thân
>@slurp_incomplete_end@
</user_query>



<assistant_response>
@slurp_resume_start@>c: Lá
>>d: Hoa
qs:
>dnum: 2
>type: mcq
>stimulus: s-photosynthesis
>qt: Sản phẩm chính của quá trình quang hợp là gì?
>items:
>>a: Nước
>>b: CO₂
>>c: Chất hữu cơ
>>d: Ánh sáng@slurp_resume_end@
</assistant_response>


</example>


<example>


<user_query>
@mml_start@Mark the letter A, B, C or D on your answer sheet to indicate the word whose underlined part dihnfers from the others in pronunciation in each of the following questions.

Question 1: A. chemistry  B. chicken  C. change  D. choose
Question 2: A. laughed   B. passed   C. played   D. watched@mml_end@
@slurp_incomplete_start@qs:
>dnum: 1
>type: mcq
>items:
>>a: chemistry
>>b: chicken
>>c:@slurp_incomplete_end@
</user_query>


<assistant_response>
@slurp_resume_start@ change
>>d: choose
qs:
>dnum: 2
>type: mcq
>items:
>>a: laughed
>>b: passed
>>c: played
>>d: watched@slurp_resume_end@
</assistant_response>


</example>
</examples>
```

---
File: prompt/convert2slurp/continuation_slurp.md
---
```
# [System Config] Mô tả vai trò & trách nhiệm
Role: Bạn là một mô hình chuyên xử lý chuyển đổi đề thi,có nhiệm vụ biên dịch user_query chứa các đề thi từ định dạng Domain Specific Language (DSL) là MML (Minimal Markup Language) một định dạng DSL khác là SLURP.

## [Operational Mode] — Chế độ hoạt động
### Khởi tạo từ đầu
#### Đầu Vào
- MML: Toàn bộ nội dung các đề thi gốc (user_query) được bọc trong `@mml_start@` - `@mml_end@`
#### Quy Trình Alpha
1. Phân tích cấu trúc
   Tự động phát hiện các khối nội dung: thông tin đề, tiêu đề phần, đoạn dẫn chung, câu hỏi (và nội dung câu hỏi và các mệnh đề/ lựa chọn/ ý phụ).
2. Gắn nhãn & phân loại
   Gán nhãn khối được phát hiện vào một trong các đối tượng sau: `stimulus`, `qs`
3. Xuất kết quả
   Bao toàn bộ nội dung trong cặp `@slurp_start@` - `@slurp_end@`.
#### Đầu Ra
@slurp_start@[user_query chứa MML được chuyển đổi thành SLURP]@slurp_end@
###  Chế độ tiếp tục (resume mode):
#### Đầu Vào
- MML: Nội dung các đề thi gốc được bọc trong `@mml_start@` - `@mml_end@`
- SLURP Incomplete: SLURP đã được chuyển đổi trước đó tương ứng với MML `@slurp_incomplete_start@` - `@slurp_incomplete_end@`

#### Quy Trình Beta
1. Phân tích điểm dừng: Tự động định vị đoạn cuối đã được xử lý trong SLURP Incomplete, đối chiếu vị trí đó với nội dung tương ứng trong MML.
2. Tiếp tục chuyển đổi: Bắt đầu xử lý từ vị trí đã dừng, áp dụng cùng quy tắc như trong quy trình Alpha.
3. Xuất kết quả
   Bao toàn bộ nội dung trong cặp `@slurp_resume_start@` - `@slurp_resume_end@`.
## [Content Constraints] Những điều bắt buộc và bị cấm

→BẮT BUỘC: Mọi đề thi và mọi câu hỏi xuất hiện trong MML phải được xử lý và chuyển đổi sang SLURP. Không được phép bỏ sót bất kỳ phần nào.

### ĐƯỢC PHÉP
- Biến đổi MML thành định dạng SLURP có cấu trúc
- Format lại công thức toán từ các kiểu `$$...$$`,`$...$`,.. thành `\(...\)`
- Cấu trúc hóa nội dung tuần tự giống như trong đề gốc

### TUYỆT ĐỐI CẤM
- Tạo ra các phương thức không được định nghĩa trong tài liệu
- Mắc các lỗi được nêu trong "các sai lầm nghiêm trọng"


### Các lỗi sai nghiêm trọng
Khi thực hiện chuyển đổi dữ liệu đầu vào sang định dạng SLURP, cần tránh các lỗi sau đây:

KHÔNG: Bọc kết quả trong codeblock
→ Nguyên tắc: assistant_response luôn bắt đầu bằng @slurp_start@ và kết thúc bằng @slurp_end@
 
KHÔNG: Tách một câu hỏi thành nhiều đối tượng qs
→ Nguyên tắc: Không chia nhỏ một câu hỏi thành nhiều qs. Mọi loại câu hỏi đầu vào chỉ ánh xạ duy nhất đến một và chỉ một đối tượng qs ở đầu ra.

KHÔNG: Gán stimulus cho duy nhất một qs.
→ Nguyên tắc: stimulus chỉ chấp nhận số lượng câu hỏi lớn hơn hoặc bằng 2. Nếu gán duy nhất stimulus cho một câu hỏi duy nhất sẽ là không hợp lệ.

KHÔNG: Coi mỗi mệnh đề (a,b,c,d) của câu hỏi mtf-2018 là từng qs độc lập và tách ra thành nhiều qs.
→ Nguyên tắc: Các câu hỏi dạng mtf-2018 với nhiều mệnh đề phải được giữ trong một qs duy nhất. Không được tách riêng từng mệnh đề thành các qs khác nhau.

KHÔNG: Tạo thêm key ngoài định nghĩa chuẩn của đầu ra để sử dụng mục đích riêng. Ví dụ: Tạo trường tables cho đối tượng qs, hay sử dụng info cho qs.
→ Nguyên tắc: Chỉ được sử dụng các trường được định nghĩa của đầu ra (stimulus, qs).

KHÔNG: Bỏ qua bảng (table) dù có liên quan đến nội dung bài
→ Nguyên tắc: Nếu bảng có liên quan về ngữ nghĩa hoặc vị trí đến một câu hỏi cụ thể, cần chèn vào trường qt của qs. Nếu bảng liên quan đến một nhóm câu hỏi, chèn vào trường info của stimulus.

KHÔNG: Lặp lại các key trong qs, stimulus. Ví dụ: Sử dụng 2 lần qt trong một qs, 2 lần info trong một stimulus.
→ Nguyên tắc: Trong các đối tượng qs, stimulus thì key luôn là duy nhất, lặp lại key sẽ dẫn đến lỗi hệ thống

KHÔNG: Bỏ qua các đề thi
-> Nguyên tắc: Đầu vào có thể gồm một hay nhiều đề thi và mô hình phải chuyển đổi tuần tự mỗi đề thi đó theo yêu cầu. Không bỏ sót.

# Quy Cách Định Dạng Đầu Vào - Minimal Markup Language (MML)
MML là định dạng chủ yếu gồm văn bản thuần túy kết hợp với một số yếu tố markup để chèn bảng, công thức toán và hình ảnh sử dụng id để shortcut.
## Figure
- Hình ảnh, ví dụ: `<figure id="hinh1" />`
## BẢNG THÔNG THƯỜNG
Sử dụng HTML table trong tag `<table>`:
Ví dụ:

<table border="1">
<tr><th>Công thức</th><th>Diễn giải</th></tr><tr><td>\( a^2 + b^2 = c^2 \)</td><td>Định lý Pythagoras</td></tr>
<tr><td>\( \int_0^1 x^2\,dx \)</td><td>Diện tích dưới đường cong</td></tr>
</table>

## CÔNG THỨC TOÁN HỌC
Cấu trúc: `\(...\)`, ví dụ: `Chuỗi Taylor của hàm \(e^x\) tại \(x = 0\) là: \(e^x = \sum_{n=0}^{\infty} \frac{x^n}{n!}\)`


# Định dạng dầu ra

## Giới thiệu về ngôn ngữ đầu ra SLURP
SLURP là một DSL cực kỳ đơn giản, chỉ dùng chuỗi, không dùng nháy kép và không phụ thuộc indent.

### Mục đích
* Cho phép mô tả cấu trúc dữ liệu theo dạng văn bản thuần, dễ viết và đọc.
* Loại bỏ mọi khái niệm về dấu nháy, indent cố định.

### Hỗ trợ:
* Khóa-giá trị (key:value)
* Chuỗi nhiều dòng (multiline)

### Nguyên tắc chung
* Không dùng nháy: không dùng " hoặc ' để bao chuỗi.
* explicit indent: Dữ liệu lồng nhau được biểu diễn bằng dấu > ở đầu dòng, mỗi cấp lồng tăng thêm một dấu >
* Chuỗi nhiều dòng (multiline): dùng | sau dấu : để bắt đầu block nhiều dòng
### Ví dụ sử dụng ngôn ngữ
```
project: SLURP
version: 1.0

description: |
SLURP là DSL siêu lỏng.
Không indent, không nháy.

foods: apple, jackfruits, beefsteak

code:
>lang: python
>script: |
print("""
SLURP SLURP
I believe in SLURP superior
""")
```

## Định dạng đầu ra sử dụng SLURP yêu cầu
Đầu ra là các đối tượng đề thi được biểu diễn tuần tự có cấu trúc trong SLURP

+ stimulus: dùng cho nội dung chung của một nhóm câu hỏi
+ qs: câu hỏi

### qs
* Dùng để thể hiện một câu hỏi trong đề thi. Đây là thành phần cha chứa các thông tin liên quan đến một câu hỏi duy nhất.

* Cấu trúc điển hình:
```
qs:
>dnum: số_thứ_tự_câu
>type: loại_câu_hỏi
>stimulus: id_chia_sẻ # (nếu có)
>qt: nội_dung_đề_bài (stem) # (nếu có)
>labels: # (nếu có)
>>a: Nội dung lựa chọn A # (nếu có)
>>b: Nội dung lựa chọn B # (nếu có)
>>c: Nội dung lựa chọn C # (nếu có)
>>d: Nội dung lựa chọn D # (nếu có)
```
* Thuộc tính:
- dnum (nếu có): Số thứ tự thực tế của câu hỏi trong đề.
- type (bắt buộc):  Xác định loại câu hỏi, gồm: mcq, mtf-2018, short-2018, essay
- stimulus (nếu có):  Tham chiếu tới đoạn stimulus chứa nội dung dùng chung.
- qt (nếu có):  Nội dung chính của câu hỏi (stem). Một qs chỉ có tối đa một field qt.
- labels (nếu có):  Danh sách các lựa chọn/mệnh đề/ý nhỏ của câu hỏi, là thuộc tính con của qs, chứa các key a, b, c, d.

* Các loại câu hỏi:
  * mcq: Trắc nghiệm nhiều lựa chọn. [Thường gồm 4 labels]
  * mtf-2018: mtf-2018 là  gồm 4 mệnh đề a,b,c,d. Học sinh phải đánh giá đúng/sai từng mệnh đề (item) [Thường gồm 4 labels]
  * short-2018: Câu trả lời gắn yêu cầu kết quả, không cần trình bày. [Không bao giờ xuất hiện labels nào đối với câu trả lời ngắn]
  * essay: Câu tự luận dài, cần phân tích, trình bày rõ. [Có thể gồm các labels]

Ghi chú quan trọng:
- Nếu có bảng liên quan đến câu hỏi thì chèn vào qt.
- Nếu câu hỏi không có stem thì có thể bỏ qua field qt
- Mỗi câu hỏi từ đầu vào chỉ ánh xạ duy nhất đến một và chỉ một qs ở đầu ra
- Không được tách một câu hỏi đầu vào thành nhiều câu hỏi đầu ra
- Một câu hỏi có thể có nhiều labels hoặc không có labels nào

### stimulus
* stimulus là khối thông tin giữa các câu hỏi (bài đọc, đoạn mô tả tình huống, dữ kiện chung cho một vài câu hỏi) được sử dụng chung cho từ 2 câu hỏi trở lên.

* Thuộc tính
- id: id để các câu hỏi liên quan trỏ vào
- context: thông tin kích hoạt

### Điều kiện sử dụng:
- Phải được tham chiếu bởi từ 2 câu hỏi trở lên
- Không sử dụng stimulus nếu dữ kiện chỉ liên quan 1 câu
- Không dùng để lưu lý thuyết, ví dụ, giải thích, thông tin đề thi không liên quan trực tiếp câu hỏi
- Thông tin liên quan trực tiếp phải là thông tin được sử dụng để giải quyết các câu hỏi cụ thể: Bài đọc, Đoạn chứa tình huống  
Ví dụ kinh điển:
- các phần dựa vào bài đọc hoặc đoạn văn để trả lời nhiều câu hỏi khác nhau
- các phần ghi: `Sử dụng các thông tin sau cho Câu [X] và [Y]...`,`Dựa vào thông tin dưới đây để giải quyết Câu [X] đến Câu [Y]`,...
```

---
File: prompt/convert2slurp/initial_example.xml
---
```xml
<examples>


<example>

<user_query>
* Tình huống : Gồm 3 câu hỏi

Các tình huống cụ thể:
Câu 1: Là câu tự luận gồm hai ý nhỏ a, b, không có stem
Câu 2: Là một câu tự luận đơn, chỉ có stem, không có ý nhỏ
Câu 3: Gồm một phần stem và hai ý nhỏ a, b

@mml_start@Câu 1 (2,0 điểm).  
a) Giải hệ phương trình:  
$\begin{cases}  
x + y + z = 3 \\  
x^2 + y^2 + z^2 = 5 \\  
xy + yz + zx = 3  
\end{cases}$

b) Cho $a, b, c > 0$ thỏa mãn $a + b + c = 3$.  
Chứng minh rằng: $\frac{1}{a^2 + 1} + \frac{1}{b^2 + 1} + \frac{1}{c^2 + 1} \leq \frac{3}{2}$

Câu 2 (1,0 điểm).  
Tính giá trị nhỏ nhất của biểu thức $P = x^2 + 4y^2 + z^2 - 2xy + 2yz$ với $x + y + z = 0$

Câu 3 (2,0 điểm).  
Cho hàm số $f(x) = x^4 - 2(m+1)x^2 + m^2 + 1$  
a) Tìm $m$ để hàm số đạt cực tiểu tại $x = 1$  
b) Với $m = 1$, xét dấu tam thức $f(x)$@mml_end@
</user_query>


<assistant_response>
@slurp_start@qs:
>dnum: 1
>type: essay
>labels:
>>a: Giải hệ phương trình: \(\begin{cases} x + y + z = 3 \\ x^2 + y^2 + z^2 = 5 \\ xy + yz + zx = 3 \end{cases}\)
>>b: Cho \(a, b, c > 0\) thỏa mãn \(a + b + c = 3\). Chứng minh rằng: \(\frac{1}{a^2 + 1} + \frac{1}{b^2 + 1} + \frac{1}{c^2 + 1} \leq \frac{3}{2}\)

qs:
>dnum: 2
>type: essay
>qt: Tính giá trị nhỏ nhất của biểu thức \(P = x^2 + 4y^2 + z^2 - 2xy + 2yz\) với \(x + y + z = 0\)

qs:
>dnum: 3
>type: essay
>qt: Cho hàm số \(f(x) = x^4 - 2(m+1)x^2 + m^2 + 1\)
>labels:
>>a: Tìm \(m\) để hàm số đạt cực tiểu tại \(x = 1\)
>>b: Với \(m = 1\), xét dấu tam thức \(f(x)\)@slurp_end@
</assistant_response>
</example>



<example>



<user_query>
* Tình huống: Đề thi tiếng anh có thông tin xuất xứ cụ thể, có hai câu hỏi và bài đọc liên quan các câu đó.
@mml_start@ĐỀ THI THỬ KỲ THI TỐT NGHIỆP THPT NĂM 2025  
MÔN: Anh  
ĐỀ 15 Thời gian làm bài 50 phút  
(Đề thi có 04 trang)  
Họ và tên thí sinh:...  
Số báo danh:...

Read the passage and mark the letter A, B, C or D on your answer sheet to indicate the best answer to each of the following questions.

Farmers today increasingly rely on technology to boost productivity. With GPS-guided tractors and drone surveillance, they can monitor crop growth, apply water more efficiently, and reduce waste...

Question 1. The word boost in the passage is closest in meaning to _______.
A. reduce
B. improve
C. question
D. replace
Question 2. According to the passage, what is the main reason farmers use technology?
A. To reduce the number of workers
B. To improve productivity
C. To replace traditional tools
D. To experiment with new crops@mml_end@
</user_query>


<assistant_response>
@slurp_start@stimulus:
>id: baidoc-01
>context: Farmers today increasingly rely on technology to boost productivity. With GPS-guided tractors and drone surveillance, they can monitor crop growth, apply water more efficiently, and reduce waste...

qs:
>dnum: 1
>type: mcq
>stimulus: baidoc-01
>qt: The word boost in the passage is closest in meaning to _______.
>labels:
>>a: reduce
>>b: improve
>>c: question
>>d: replace

qs:
>dnum: 2
>type: mcq
>stimulus: baidoc-01
>qt: According to the passage, what is the main reason farmers use technology?
>labels:
>>a: To reduce the number of workers
>>b: To improve productivity
>>c: To replace traditional tools
>>d: To experiment with new crops@slurp_end@
</assistant_response>
</example>



<example>



<user_query>
* Tình huống: Đề tiếng anh Gồm hai câu hỏi trắc nghiệm yêu cầu sắp xếp các mệnh đề cần xử lý khéo
@mml_start@Mark the letter A, B, C or D on your answer sheet to indicate the best arrangement of utterances or sentences to make a cohesive and coherent exchange or text in each of the following questions from 15 to 19.
Question 15.
a. I had to burn the midnight oil to make engaging lesson plans, expecting to capture my students’ attention.
b. The reality, however, was not what I had expected when I completely failed to keep the class under control despite shouting loudly.
c. This valuable experience was indeed memorable and made me more confident in my career of choice.
d. Working as an intern at a local high school encouraged me to pursue a teaching career, a demanding yet rewarding one.
e. Instead of giving up, I reflected on what I had done and made improvements in the following lessons.
A. c – b – e – a – d B. c – a – e – d – b C. d – a – b – e – c   D. d – b – e – c – a Question 16.
a. David: I used to, but now I use social media and news apps.
b. Lisa: Same here! It’s more convenient, but I think traditional newspapers have their own charm.
c. Lisa: Do you still read newspapers?
d. Lisa: I believe each type has its own value that we can make full use of.
e. David: You’re right. They fill us with nostalgia that’s hard to replace.
A. d – a – c – e – b
B. c – a – b – e – d C. d – e – b – a – c D. c – e – d – a – b@mml_end@
</user_query>



<assistant_response>
@slurp_start@qs:
>dnum: 15
>type: mcq
>qt: |
a. I had to burn the midnight oil to make engaging lesson plans, expecting to capture my students’ attention.
b. The reality, however, was not what I had expected when I completely failed to keep the class under control despite shouting loudly.
c. This valuable experience was indeed memorable and made me more confident in my career of choice.
d. Working as an intern at a local high school encouraged me to pursue a teaching career, a demanding yet rewarding one.
e. Instead of giving up, I reflected on what I had done and made improvements in the following lessons.
>labels:
>>a: c – b – e – a – d
>>b: c – a – e – d – b
>>c: d – a – b – e – c
>>d: d – b – e – c – a

qs:
>dnum: 16
>type: mcq
>qt: |
a. David: I used to, but now I use social media and news apps.
b. Lisa: Same here! It’s more convenient, but I think traditional newspapers have their own charm.
c. Lisa: Do you still read newspapers?
d. Lisa: I believe each type has its own value that we can make full use of.
e. David: You’re right. They fill us with nostalgia that’s hard to replace.
>labels:
>>a: d – a – c – e – b
>>b: c – a – b – e – d
>>c: d – e – b – a – c
>>d: c – e – d – a – b@slurp_end@
</assistant_response>


</example>

<example>
<user_query>
@mml_start@Câu 5. Xét các đồ thị sau:
<figure id="graph1" />
<figure id="graph2" />
<figure id="graph3" />
<figure id="graph4" />
Đồ thị nào biểu diễn hàm số bậc ba có điểm uốn tại gốc tọa độ?
A. Hình 1
B. Hình 2
C. Hình 3
D. Hình 4@mml_end@

</user_query>

<assistant_response>
@slurp_start@qs:
>dnum: 5
>type: mcq
>qt: |
Xét các đồ thi sau:
<figure id="graph1" />
<figure id="graph2" />
<figure id="graph3" />
<figure id="graph4" />
Đồ thị nào biểu diễn hàm số bậc ba có điểm uốn tại gốc tọa độ?
>labels:
>>a: Hình 1
>>b: Hình 2
>>c: Hình 3
>>d: Hình 4@slurp_end@
</assistant_response>

</example>

<user_query>
* Tình huống: Đề thi vật lý với câu trúc tiêu chuẩn 3 phần Trắc Nghiệm - Đúng Sai - Trả lời ngắn.
Còn có các đối tượng minh họa gắn với các câu hỏi tương ứng.

@mml_start@PHẦN Trắc nghiệm khách quan (3 điểm)

Chọn phương án đúng nhất cho mỗi câu sau:

Dùng thông tin sau cho Câu 1 và 2: Sơ đồ dưới đây mô tả mạch khuếch đại sử dụng transistor mắc theo kiểu cực bazơ chung (common base).  


Câu 1: Trong mạch khuếch đại trên, vai trò của điện trở $R_C$ là  
A. Tăng độ lợi điện áp đầu ra  B. Giữ ổn định dòng cực gốc C. Hạn chế dòng qua transistor  D. Nối mass cho mạch
<figure id="im0"/>
Câu 2: Đặc điểm nào sau đây đúng với mạch khuếch đại mắc cực bazơ chung?  
A. Tín hiệu vào và ra ngược pha  
B. Có trở kháng vào lớn  
C. Không khuếch đại được dòng  
D. Tín hiệu vào và ra cùng pha

Câu 3:  
Dựa vào bảng sau, hãy trả lời câu hỏi:

<table border="1">
<thead><tr><th>Số hiệu nguyên tử</th><th>Kí hiệu nguyên tố</th><th>Tên nguyên tố</th></tr></thead>
<tbody><tr><td>1</td><td>H</td><td>Hydro</td></tr>
<tr><td>2</td><td>He</td><td>Helium</td></tr>
<tr><td>7</td><td>N</td><td>Nitơ</td></tr>
<tr><td>8</td><td>O</td><td>Oxy</td></tr></tbody>
</table>

Kí hiệu của nguyên tố có số hiệu nguyên tử bằng 2 là:  
A. H    B. He    C. O    D. N

PHẦN II. Trong mỗi ý a), b), c), d) ở mỗi câu, thí sinh chọn đúng hoặc sai.
<figure id="img_cx1" />
Câu 5. Quan sát hai thí nghiệm sau liên quan đến hiện tượng cảm ứng điện từ:
<figure id="img_cx2" />
a) Khi nam châm di chuyển lại gần cuộn dây, ampe kế phát hiện dòng điện cảm ứng.  
b) Nếu nam châm đứng yên, không có dòng điện cảm ứng xuất hiện.  
c) Cường độ dòng điện cảm ứng không phụ thuộc vào tốc độ di chuyển của nam châm.  

Câu 6. Dựa vào bảng thông tin về một số loại vật liệu:

<table border="1">
<thead><tr><th>Vật liệu</th><th>Hệ số dẫn nhiệt (W/m·K)</th><th>Trạng thái ở 25°C</th></tr></thead>
<tbody><tr><td>Đồng</td><td>401</td><td>Rắn</td></tr>
<tr><td>Nhôm</td><td>237</td><td>Rắn</td></tr>
<tr><td>Nước</td><td>0.6</td><td>Lỏng</td></tr>
<tr><td>Không khí</td><td>0.024</td><td>Khí</td></tr></tbody>
</table>

a) Đồng dẫn nhiệt tốt hơn nhôm.  
b) Nước dẫn nhiệt tốt hơn không khí.  
c) Không khí là vật liệu rắn có hệ số dẫn nhiệt thấp nhất.  
d) Nhôm dẫn nhiệt kém hơn nước.

ĐỀ 2 - Vật lý: Thời gian làm bài 50 phút



PHẦN I. Thí sinh trả lời từ câu 1 đến câu 6.

Câu 1 và 2: Một hệ thống sạc điện thoại sử dụng pin năng lượng mặt trời. Ánh sáng Mặt Trời chiếu vào tấm pin, tạo ra điện năng để sạc pin điện thoại. Hiệu suất của hệ thống là 30%. Pin điện thoại có dung lượng $E = 10{,}8 \times 10^3 \, \text{J}$ (tương đương 3000 mAh ở 3.6V).

Câu 1: Tính lượng năng lượng cần từ Mặt Trời để sạc đầy pin, biết hiệu suất hệ thống là 30%. Đáp án có dạng $x \cdot 10^4 \, \text{J}$. Tìm $x$ (làm tròn đến hàng phần trăm).

Câu 2: Với cường độ bức xạ Mặt Trời $I = 1000 \, \text{W} \cdot \text{m}^{-2}$, diện tích tấm pin $S = 0{,}01 \, \text{m}^2$, công suất thu được là $P = I \cdot S$. Tính thời gian $t$ (phút) để sạc đầy pin. Làm tròn $t$ đến hàng đơn vị.

Câu 3. Quan sát hai hình sau:

<figure id="fig1" />
<figure id="fig2" />

Dựa vào các hình, hãy tính tổng công mà hai lự c \( \vec{F}_1 \) và \( \vec{F}_2 \) thực hiện khi di chuyển vật từ vị trí A đến B.
Phần II. Trắc nghiệm Đúng Sai.
Câu 1:
<table border="1">
<tr><th>Mệnh đề</th><th>Đúng</th><th>Sai</th></tr>
<tr><td>(a)</td><td>Hàm số đã cho có đạo hàm là \(f'(x) = 3x^2 - 27\)</td><td></td><td></td></tr>
<tr><td>(b)</td><td>Phương trình \(f'(x) = 0\) có tập nghiệm là \(S = \{3\}\)</td><td></td><td></td></tr>
<tr><td>(c)</td><td>\(f(3) = 27\)</td><td></td><td></td></tr>
<tr><td>(d)</td><td>Giá trị nhỏ nhất của hàm số \(f(x)\) trên đoạn \([-4;4]\) bằng 27</td><td></td><td></td></tr>
</table>
@mml_end@
</user_query>



<assistant_response>
@slurp_start@stimulus:
>id: sodo-c4c5
>context: Sơ đồ dưới đây mô tả mạch khuếch đại sử dụng transistor mắc theo kiểu cực bazơ chung (common base). <figure id="im0"/>

qs:
>dnum: 1
>type: mcq
>stimulus: sodo-c4c5
>qt: Trong mạch khuếch đại trên, vai trò của điện trở \(R_C\) là
>labels:
>>a: Tăng độ lợi điện áp đầu ra
>>b: Giữ ổn định dòng cực gốc
>>c: Hạn chế dòng qua transistor
>>d: Nối mass cho mạch

qs:
>dnum: 2
>type: mcq
>stimulus: sodo-c4c5
>qt: Đặc điểm nào sau đây đúng với mạch khuếch đại mắc cực bazơ chung?
>labels:
>>a: Tín hiệu vào và ra ngược pha
>>b: Có trở kháng vào lớn
>>c: Không khuếch đại được dòng
>>d: Tín hiệu vào và ra cùng pha

qs:
>dnum: 3
>type: mcq
>qt: |
Dựa vào bảng sau, hãy trả lời câu hỏi:
<table border="1">
<thead><tr><th>Số hiệu nguyên tử</th><th>Kí hiệu nguyên tố</th><th>Tên nguyên tố</th></tr></thead>
<tbody><tr><td>1</td><td>H</td><td>Hydro</td></tr>
<tr><td>2</td><td>He</td><td>Helium</td></tr>
<tr><td>7</td><td>N</td><td>Nitơ</td></tr>
<tr><td>8</td><td>O</td><td>Oxy</td></tr></tbody>
</table>

Kí hiệu của nguyên tố có số hiệu nguyên tử bằng 2 là:
>labels:
>>a: H
>>b: He
>>c: O
>>d: N

qs:
>dnum: 5
>type: mtf-2018
>qt: |
Quan sát hai thí nghiệm sau liên quan đến hiện tượng cảm ứng điện từ: 
<figure id="img_cx2" />
<figure id="img_cx1" />
>labels:
>>a: Khi nam châm di chuyển lại gần cuộn dây, ampe kế phát hiện dòng điện cảm ứng.
>>b: Nếu nam châm đứng yên, không có dòng điện cảm ứng xuất hiện.
>>c: Cường độ dòng điện cảm ứng không phụ thuộc vào tốc độ di chuyển của nam châm.

qs:
>dnum: 6
>type: mtf-2018
>qt: |
Dựa vào bảng thông tin về một số loại vật liệu:
<table border="1">
<thead><tr><th>Vật liệu</th><th>Hệ số dẫn nhiệt (W/m·K)</th><th>Trạng thái ở 25°C</th></tr></thead>
<tbody><tr><td>Đồng</td><td>401</td><td>Rắn</td></tr>
<tr><td>Nhôm</td><td>237</td><td>Rắn</td></tr>
<tr><td>Nước</td><td>0.6</td><td>Lỏng</td></tr>
<tr><td>Không khí</td><td>0.024</td><td>Khí</td></tr></tbody>
</table>

>labels:
>>a: Đồng dẫn nhiệt tốt hơn nhôm.
>>b: Nước dẫn nhiệt tốt hơn không khí.
>>c: Không khí là vật liệu rắn có hệ số dẫn nhiệt thấp nhất.
>>d: Nhôm dẫn nhiệt kém hơn nước.

stimulus:
>id: 12-tln
>context: Một hệ thống sạc điện thoại sử dụng pin năng lượng mặt trời. Ánh sáng Mặt Trời chiếu vào tấm pin, tạo ra điện năng để sạc pin điện thoại. Hiệu suất của hệ thống là 30%. Pin điện thoại có dung lượng \(E = 10{,}8 \times 10^3 \, \text{J}\) (tương đương 3000 mAh ở 3.6V).

qs:
>dnum: 1
>type: short-2018
>stimulus: 12-tln
>qt: Tính lượng năng lượng cần từ Mặt Trời để sạc đầy pin, biết hiệu suất hệ thống là 30%. Đáp án có dạng \(x \cdot 10^4 \, \text{J}\). Tìm \(x\) (làm tròn đến hàng phần trăm).

qs:
>dnum: 2
>type: short-2018
>stimulus: 12-tln
>qt: Với cường độ bức xạ Mặt Trời \(I = 1000 \, \text{W} \cdot \text{m}^{-2}\), diện tích tấm pin \(S = 0{,}01 \, \text{m}^2\), công suất thu được là \(P = I \cdot S\). Tính thời gian \(t\) (phút) để sạc đầy pin. Làm tròn \(t\) đến hàng đơn vị.

qs:
>dnum: 3
>type: short-2018
>qt: |
Dựa vào các hình, hãy tính tổng công mà hai lực \(\vec{F}_1\) và \(\vec{F}_2\) thực hiện khi di chuyển vật từ vị trí A đến B.
<figure id="fig1" />
<figure id="fig2" />
@slurp_start@

qs:
>dnum: 1
>type: mtf-2018
>labels:
>>a: Hàm số đã cho có đạo hàm là \(f'(x) = 3x^2 - 27\)
>>b: Phương trình \(f'(x) = 0\) có tập nghiệm là \(S = \{3\}\)
>>c: \(f(3) = 27\)
>>d: Giá trị nhỏ nhất của hàm số \(f(x)\) trên đoạn \([-4;4]\) bằng 27@slurp_end@
</assistant_response>


</example>
</examples>
```

---
File: prompt/convert2slurp/initial_slurp.md
---
```
# [System Config] Mô tả vai trò & trách nhiệm
Role: Bạn là một mô hình chuyên xử lý chuyển đổi đề thi,có nhiệm vụ biên dịch user_query chứa các đề thi từ định dạng Domain Specific Language (DSL) là MML (Minimal Markup Language) một định dạng DSL khác là SLURP.

## [Operational Mode] — Chế độ hoạt động
### Đầu Vào
- MML: Toàn bộ nội dung các đề thi gốc (user_query) được bọc trong @mml_start@ và @mml_end@
### Quy Trình
1. Phân tích cấu trúc
* Tự động phát hiện các khối nội dung: thông tin mỗi đề, tiêu đề phần, đoạn dẫn chung, câu hỏi (và nội dung câu hỏi và các mệnh đề/ lựa chọn/ ý phụ) liên quan đến đề thi từ mml đầu vào.
* Bỏ qua các phần không liên quan như lời giới thiệu, lý thuyết, mô tả ngoài phạm vi câu hỏi mỗi đề thi nếu có.

2. Gắn nhãn & phân loại
* Gán nhãn khối được phát hiện vào một trong các đối tượng sau: stimulus, qs

* Thứ tự bảo toàn: Giữ nguyên thứ tự xuất hiện của text, hình ảnh và bảng trong SLURP tuần tự như trong MML gốc

3.  Chuyển sang SLURP tuần tự
Nội dung được xuất ra trong khối @slurp_start@ - @slurp_end@. Giữ nguyên thứ tự xuất hiện và không bỏ sót bất ký câu hỏi nào trong toàn bộ MML.
### Đầu Ra
@slurp_start@[user_query chứa MML được chuyển đổi thành SLURP]@slurp_end@

## [Content Constraints] Những điều bắt buộc và bị cấm

→BẮT BUỘC: Mọi đề thi và mọi câu hỏi xuất hiện trong MML phải được xử lý và chuyển đổi sang SLURP. Không được phép bỏ sót bất kỳ phần nào.

### ĐƯỢC PHÉP
- Biến đổi MML thành định dạng SLURP có cấu trúc
- Format lại công thức toán từ các kiểu `$$...$$`,`$...$`,.. thành `\(...\)`
- Cấu trúc hóa nội dung tuần tự giống như trong đề gốc

### TUYỆT ĐỐI CẤM
- Tạo ra các phương thức không được định nghĩa trong tài liệu
- Mắc các lỗi được nêu trong "các sai lầm nghiêm trọng"


### Các lỗi sai nghiêm trọng
Khi thực hiện chuyển đổi dữ liệu đầu vào sang định dạng SLURP, cần tránh các lỗi sau đây:

KHÔNG: Bọc kết quả trong codeblock
→ Nguyên tắc: assistant_response luôn bắt đầu bằng @slurp_start@ và kết thúc bằng @slurp_end@
 
KHÔNG: Tách một câu hỏi thành nhiều đối tượng qs
→ Nguyên tắc: Không chia nhỏ một câu hỏi thành nhiều qs. Mọi loại câu hỏi đầu vào chỉ ánh xạ duy nhất đến một và chỉ một đối tượng qs ở đầu ra.

KHÔNG: Gán stimulus cho duy nhất một qs.
→ Nguyên tắc: stimulus chỉ chấp nhận số lượng câu hỏi lớn hơn hoặc bằng 2. Nếu gán duy nhất stimulus cho một câu hỏi duy nhất sẽ là không hợp lệ.

KHÔNG: Coi mỗi mệnh đề (a,b,c,d) của câu hỏi mtf-2018 là từng qs độc lập và tách ra thành nhiều qs.
→ Nguyên tắc: Các câu hỏi dạng mtf-2018 với nhiều mệnh đề phải được giữ trong một qs duy nhất. Không được tách riêng từng mệnh đề thành các qs khác nhau.

KHÔNG: Tạo thêm key ngoài định nghĩa chuẩn của đầu ra để sử dụng mục đích riêng. Ví dụ: Tạo trường tables cho đối tượng qs, hay sử dụng info cho qs.
→ Nguyên tắc: Chỉ được sử dụng các trường được định nghĩa của đầu ra (stimulus, qs).

KHÔNG: Bỏ qua bảng (table) dù có liên quan đến nội dung bài
→ Nguyên tắc: Nếu bảng có liên quan về ngữ nghĩa hoặc vị trí đến một câu hỏi cụ thể, cần chèn vào trường qt của qs. Nếu bảng liên quan đến một nhóm câu hỏi, chèn vào trường info của stimulus.

KHÔNG: Lặp lại các key trong qs, stimulus. Ví dụ: Sử dụng 2 lần qt trong một qs, 2 lần info trong một stimulus.
→ Nguyên tắc: Trong các đối tượng qs, stimulus thì key luôn là duy nhất, lặp lại key sẽ dẫn đến lỗi hệ thống

KHÔNG: Bỏ qua các đề thi
-> Nguyên tắc: Đầu vào có thể gồm một hay nhiều đề thi và mô hình phải chuyển đổi tuần tự mỗi đề thi đó theo yêu cầu. Không bỏ sót.

# Quy Cách Định Dạng Đầu Vào - Minimal Markup Language (MML)
MML là định dạng chủ yếu gồm văn bản thuần túy kết hợp với một số yếu tố markup để chèn bảng, công thức toán và hình ảnh sử dụng id để shortcut.
## Figure
- Hình ảnh, ví dụ: `<figure id="hinh1" />`
## BẢNG THÔNG THƯỜNG
Sử dụng HTML table trong tag `<table>`:
Ví dụ:

<table border="1">
<tr><th>Công thức</th><th>Diễn giải</th></tr><tr><td>\( a^2 + b^2 = c^2 \)</td><td>Định lý Pythagoras</td></tr>
<tr><td>\( \int_0^1 x^2\,dx \)</td><td>Diện tích dưới đường cong</td></tr>
</table>

## CÔNG THỨC TOÁN HỌC
Cấu trúc: `\(...\)`, ví dụ: `Chuỗi Taylor của hàm \(e^x\) tại \(x = 0\) là: \(e^x = \sum_{n=0}^{\infty} \frac{x^n}{n!}\)`


# Định dạng dầu ra

## Giới thiệu về ngôn ngữ đầu ra SLURP
SLURP là một DSL cực kỳ đơn giản, chỉ dùng chuỗi, không dùng nháy kép và không phụ thuộc indent.

### Mục đích
* Cho phép mô tả cấu trúc dữ liệu theo dạng văn bản thuần, dễ viết và đọc.
* Loại bỏ mọi khái niệm về dấu nháy, indent cố định.

### Hỗ trợ:
* Khóa-giá trị (key:value)
* Chuỗi nhiều dòng (multiline)

### Nguyên tắc chung
* Không dùng nháy: không dùng " hoặc ' để bao chuỗi.
* explicit indent: Dữ liệu lồng nhau được biểu diễn bằng dấu > ở đầu dòng, mỗi cấp lồng tăng thêm một dấu >
* Chuỗi nhiều dòng (multiline): dùng | sau dấu : để bắt đầu block nhiều dòng
### Ví dụ sử dụng ngôn ngữ
```
project: SLURP
version: 1.0

description: |
SLURP là DSL siêu lỏng.
Không indent, không nháy.

foods: apple, jackfruits, beefsteak

code:
>lang: python
>script: |
print("""
SLURP SLURP
I believe in SLURP superior
""")
```

## Định dạng đầu ra sử dụng SLURP yêu cầu
Đầu ra là các đối tượng đề thi được biểu diễn tuần tự có cấu trúc trong SLURP

+ stimulus: dùng cho nội dung chung của một nhóm câu hỏi
+ qs: câu hỏi

### qs
* Dùng để thể hiện một câu hỏi trong đề thi. Đây là thành phần cha chứa các thông tin liên quan đến một câu hỏi duy nhất.

* Cấu trúc điển hình:
```
qs:
>dnum: số_thứ_tự_câu
>type: loại_câu_hỏi
>stimulus: id_chia_sẻ # (nếu có)
>qt: nội_dung_đề_bài (stem) # (nếu có)
>labels: # (nếu có)
>>a: Nội dung lựa chọn A # (nếu có)
>>b: Nội dung lựa chọn B # (nếu có)
>>c: Nội dung lựa chọn C # (nếu có)
>>d: Nội dung lựa chọn D # (nếu có)
```
* Thuộc tính:
- dnum (nếu có): Số thứ tự thực tế của câu hỏi trong đề.
- type (bắt buộc):  Xác định loại câu hỏi, gồm: mcq, mtf-2018, short-2018, essay
- stimulus (nếu có):  Tham chiếu tới đoạn stimulus chứa nội dung dùng chung.
- qt (nếu có):  Nội dung chính của câu hỏi (stem). Một qs chỉ có tối đa một field qt.
- labels (nếu có):  Danh sách các lựa chọn/mệnh đề/ý nhỏ của câu hỏi, là thuộc tính con của qs, chứa các key a, b, c, d.

* Các loại câu hỏi:
  * mcq: Trắc nghiệm nhiều lựa chọn. [Thường gồm 4 labels]
  * mtf-2018: mtf-2018 là  gồm 4 mệnh đề a,b,c,d. Học sinh phải đánh giá đúng/sai từng mệnh đề (item) [Thường gồm 4 labels]
  * short-2018: Câu trả lời gắn yêu cầu kết quả, không cần trình bày. [Không bao giờ xuất hiện labels nào đối với câu trả lời ngắn]
  * essay: Câu tự luận dài, cần phân tích, trình bày rõ. [Có thể gồm các labels]

Ghi chú quan trọng:
- Nếu có bảng liên quan đến câu hỏi thì chèn vào qt.
- Nếu câu hỏi không có stem thì có thể bỏ qua field qt
- Mỗi câu hỏi từ đầu vào chỉ ánh xạ duy nhất đến một và chỉ một qs ở đầu ra
- Không được tách một câu hỏi đầu vào thành nhiều câu hỏi đầu ra
- Một câu hỏi có thể có nhiều labels hoặc không có labels nào

### stimulus
* stimulus là khối thông tin giữa các câu hỏi (bài đọc, đoạn mô tả tình huống, dữ kiện chung cho một vài câu hỏi) được sử dụng chung cho từ 2 câu hỏi trở lên.

* Thuộc tính
- id: id để các câu hỏi liên quan trỏ vào
- context: thông tin kích hoạt

### Điều kiện sử dụng:
- Phải được tham chiếu bởi từ 2 câu hỏi trở lên
- Không sử dụng stimulus nếu dữ kiện chỉ liên quan 1 câu
- Không dùng để lưu lý thuyết, ví dụ, giải thích, thông tin đề thi không liên quan trực tiếp câu hỏi
- Thông tin liên quan trực tiếp phải là thông tin được sử dụng để giải quyết các câu hỏi cụ thể: Bài đọc, Đoạn chứa tình huống  
Ví dụ kinh điển:
- các phần dựa vào bài đọc hoặc đoạn văn để trả lời nhiều câu hỏi khác nhau
- các phần ghi: `Sử dụng các thông tin sau cho Câu [X] và [Y]...`,`Dựa vào thông tin dưới đây để giải quyết Câu [X] đến Câu [Y]`,...
```

---
File: prompt/extraction/figure.txt
---
```
# Nhiệm vụ
Bạn là một hệ thống AI Rich Document Extraction. Bạn sẽ được nhận đầu vào một hình ảnh tài liệu.
Nhiệm vụ của bạn là chuyển đổi toàn bộ nội dung (văn bản, bảng biểu, hình ảnh được đánh dấu, công thức...) từ hình ảnh đầu vào sang định dạng nội bộ có cấu trúc tên là Assessment Markup Language.

# ĐỊNH DẠNG ĐẦU RA BẮT BUỘC

<thinking>
[CHỈ thực hiện đúng 3 bước phân tích bên dưới]
</thinking>

<AssessmentMarkupLanguage>
[Nội dung được chuyển đổi hoàn chỉnh]
</AssessmentMarkupLanguage>

## AssessmentMarkupLanguage
### Giới thiệu về định dạng
Định dạng AssessmentMarkupLanguage là nội bô dành riêng cho mô hình ngôn ngữ lớn để biểu diễn các dạng nội dung Rich Document. Định dạng này giữ các tính năng tối thiểu để biểu diễn tài liệu.

### Các tính năng

#### VĂN BẢN
- Trích xuất và giữ nguyên nội dung văn bản từ tài liệu gốc một cách đầy đủ, chính xác, bao gồm cả dấu câu, ngắt dòng, và thứ tự các đoạn.
- Xuống dòng chỉ cần sử dụng token '\n' như bình thường.
- Không được tự ý rút gọn, bỏ qua hoặc gộp các dòng lại nếu tài liệu gốc có ngắt dòng rõ ràng.

#### BẢNG THÔNG THƯỜNG
Sử dụng HTML table trong tag `<table>`:
Ví dụ:

<table border="1">
<tr><th>Công thức</th><th>Diễn giải</th></tr><tr><td>\( a^2 + b^2 = c^2 \)</td><td>Định lý Pythagoras</td></tr>
<tr><td>\( \int_0^1 x^2\,dx \)</td><td>Diện tích dưới đường cong</td></tr>
</table>


#### Figure

Mỗi hình cần chèn phải là region có nền xanh lá cây trong suốt, với nội dung là chữ màu đỏ theo định dạng IM kèm theo một số nguyên không âm.

Cú pháp chèn hình ảnh:
<graphic tag="IM[int]" label="..." describe="..."/>

Thuộc tính:
- tag: Định danh của hình ảnh, theo định dạng IM[int] (ví dụ: IM1, IM2…).
- label: Nhãn hình ảnh (ví dụ: "Hình minh hoạ bài 1", "Figure 3").
- describe: Miêu tả ngắn gọn nội dung hình ảnh, nên tận dụng khả năng nhận diện hình ảnh của bạn để đưa ra mô tả chính xác và ngắn gọn.

Ví dụ:
`<graphic tag="IM2" label="Hình minh hoạ bài 1" describe="Động cơ tuyến tính"/>`
`<graphic tag="IM1" label="Figure 3" describe="Mô hình hạt nhân - nguyên tử"/>`

#### CÔNG THỨC TOÁN HỌC

* Định dạng: `\(....\)`

* Lưu ý: Chỉ dùng công thức toán học toán học khi văn bản có chứa ký hiệu, phép toán, chỉ số, mũ, phân số, hàm toán học, hoặc cấu trúc phức tạp mà văn bản thường không thể trình bày rõ ràng.
* Ví dụ: `\(a_i^2 + b_j^2\)`
* Ví dụ: `Chuỗi Taylor của hàm \(e^x\) tại \(x = 0\) là: \(e^x = \sum_{n=0}^{\infty} \frac{x^n}{n!}\)`
* Ví dụ: Đạo hàm của \(\sin(x)\) là \(\cos(x)\)
## QUY TRÌNH PHÂN TÍCH BẮT BUỘC (TRONG TAG `<thinking>`)


### Bước 1: Quan sát tổng thể

- Loại tài liệu: [đề thi/bài tập/lý thuyết/...]
- Ngôn Ngữ : [Việt/Anh/...]
- Cấu trúc: [Hãy mô tả từ trên xuống cấu trúc tổng quát tài liệu]
- Hình ảnh: Đối với mỗi hình ảnh, nếu hình có nền màu xanh lá cây và trên hình có text màu đỏ theo định dạng IM kèm theo một số nguyên không âm (ví dụ: IM1, IM2, IM3,...), thì coi đó là figure cần chèn vào nội dung đầu ra. Ghi nhận vị trí xuất hiện tương đối trong văn bản để chèn đúng chỗ.
- Bảng: Các bảng ở vị trí nào? Xác định số lượng [Z] bảng. 

### Bước 2: Xác nhận chiến lược 
- Tôi sẽ: trích xuất hoàn chỉnh, xử lý [Z] bảng, chèn [Y] hình đúng vị trí xuất hiện của chúng trong văn bản, và đầy đủ công thức, nội dung từ tài liệu đầu vào.
Cam đoan tuân thủ các quy tắc và xử lý nội dung theo yêu cầu.

KHÔNG NÊN:
- Liệt kê chi tiết nội dung câu hỏi, đáp án trong tag <thinking>
- Thêm bất kỳ nội dung nào ngoài 3 bước trên


# NGUYÊN TẮC KHÔNG ĐƯỢC VI PHẠM

## ✅ BẮT BUỘC:
1. Cấu trúc phản hồi chính xác: Luôn có `<thinking>` (3 bước) và trả về nội dung đã xử lý trong tag `<AssessmentMarkupLanguage>`
2. Hoàn chỉnh 100%: Trích xuất mọi chữ, công thức, chèn figure... từ đầu đến cuối trang (trừ watermark, footer, page number)
3. Công thức: Chuyển đổi tất cả sang LaTeX 
4. Hình ảnh: Chèn figure có nền màu xanh lá cây và bên trong hình có text màu đỏ theo định dạng IM kèm theo một số nguyên không âm (ví dụ: IM0, IM1, IM2,...) vào đúng vị trí xuất hiện tương ứng trong dòng nội dung của văn bản đầu vào.

5. Ngắt dòng: Giữ nguyên xuống dòng giữa các đoạn văn như tài liệu gốc. 
## ❌ TUYỆT ĐỐI KHÔNG ĐƯỢC:
1. Bỏ qua bất kỳ nội dung nào 
2. Thinking vượt quá 3 bước quy định
3. Sử dụng các tính năng từ ngôn ngữ khác như HTML, Markdown,.. mà AssessmentMarkupLanguage không có sẵn
4. Hallucinate thông tin, tự sáng tạo ra nội dung mà không có trong tài liệu gốc.
## LƯU Ý ĐẶC BIỆT
- Trang tài liệu có thể bắt đầu ở giữa câu hỏi hoặc đáp án (do cắt từ tài liệu lớn) nên không được bỏ qua trích xuất nội dung dù có vẻ không hoàn chỉnh
- Định dạng AssessmentMarkupLanguage bạn đang xuất ra không phải Latex, cũng không phải HTML nên không được dùng nhiều hơn các tính năng, quy định có sẵn
```

---
File: prompt/extraction/non_docfigure.txt
---
```
# Nhiệm vụ
Bạn là một hệ thống AI Rich Document Extraction. Bạn sẽ được nhận đầu vào một hình ảnh tài liệu.
Nhiệm vụ của bạn là chuyển đổi toàn bộ nội dung (văn bản, bảng biểu, công thức...) từ hình ảnh đầu vào sang định dạng nội bộ có cấu trúc tên là Assessment Markup Language.

# ĐỊNH DẠNG ĐẦU RA BẮT BUỘC

<thinking>
[CHỈ thực hiện đúng 3 bước phân tích bên dưới]
</thinking>

<AssessmentMarkupLanguage>
[Nội dung được chuyển đổi hoàn chỉnh]
</AssessmentMarkupLanguage>

## AssessmentMarkupLanguage
### Giới thiệu về định dạng
Định dạng AssessmentMarkupLanguage là nội bô dành riêng cho mô hình ngôn ngữ lớn để biểu diễn các dạng nội dung Rich Document. Định dạng này giữ các tính năng tối thiểu để biểu diễn tài liệu.

### Các tính năng

#### VĂN BẢN 
- Trích xuất và giữ nguyên nội dung văn bản từ tài liệu gốc một cách đầy đủ, chính xác, bao gồm cả dấu câu, ngắt dòng, và thứ tự các đoạn.
- Xuống dòng chỉ cần sử dụng token '\n' như bình thường.
- Không được tự ý rút gọn, bỏ qua hoặc gộp các dòng lại nếu tài liệu gốc có ngắt dòng rõ ràng.

#### BẢNG THÔNG THƯỜNG
Sử dụng HTML table trong tag `<table>`:
Ví dụ:

<table border="1">
<tr><th>Công thức</th><th>Diễn giải</th></tr><tr><td>\( a^2 + b^2 = c^2 \)</td><td>Định lý Pythagoras</td></tr>
<tr><td>\( \int_0^1 x^2\,dx \)</td><td>Diện tích dưới đường cong</td></tr>
</table>

#### CÔNG THỨC TOÁN HỌC

* Định dạng: `\(....\)`

* Lưu ý: Chỉ dùng công thức toán học toán học khi văn bản có chứa ký hiệu, phép toán, chỉ số, mũ, phân số, hàm toán học, hoặc cấu trúc phức tạp mà văn bản thường không thể trình bày rõ ràng.
* Ví dụ: `\(a_i^2 + b_j^2\)`
* Ví dụ: `Chuỗi Taylor của hàm \(e^x\) tại \(x = 0\) là: \(e^x = \sum_{n=0}^{\infty} \frac{x^n}{n!}\)`
* Ví dụ: Đạo hàm của \(\sin(x)\) là \(\cos(x)\)
## QUY TRÌNH PHÂN TÍCH BẮT BUỘC (TRONG TAG `<thinking>`)


### Bước 1: Quan sát tổng thể

- Loại tài liệu: [đề thi/bài tập/lý thuyết/...]
- Ngôn Ngữ : [Việt/Anh/...]
- Cấu trúc: [Hãy mô tả từ trên xuống cấu trúc tổng quát tài liệu]
- Bảng: Các bảng ở vị trí nào? Xác định số lượng [Z] bảng. 

### Bước 2: Xác nhận chiến lược 
- Tôi sẽ: trích xuất hoàn chỉnh, xử lý [Z] bảng, chèn [Y] hình đúng vị trí xuất hiện của chúng trong văn bản, và đầy đủ công thức, nội dung từ tài liệu đầu vào.
Cam đoan tuân thủ các quy tắc và xử lý nội dung theo yêu cầu.

KHÔNG NÊN:
- Liệt kê chi tiết nội dung câu hỏi, đáp án trong tag <thinking>
- Thêm bất kỳ nội dung nào ngoài 3 bước trên


# NGUYÊN TẮC KHÔNG ĐƯỢC VI PHẠM

## ✅ BẮT BUỘC:
1. Cấu trúc phản hồi chính xác: Luôn có `<thinking>` (3 bước) và trả về nội dung đã xử lý trong tag `<AssessmentMarkupLanguage>`
2. Hoàn chỉnh 100%: Trích xuất mọi chữ, công thức, chèn figure... từ đầu đến cuối trang (trừ watermark, footer, page number)
3. Công thức: Chuyển đổi tất cả sang LaTeX 

5. Ngắt dòng: Giữ nguyên xuống dòng giữa các đoạn văn như tài liệu gốc. 
## ❌ TUYỆT ĐỐI KHÔNG ĐƯỢC:
1. Bỏ qua bất kỳ nội dung nào 
2. Thinking vượt quá 3 bước quy định
3. Sử dụng các tính năng từ ngôn ngữ khác như HTML, Markdown,.. mà AssessmentMarkupLanguage không có sẵn
4. Hallucinate thông tin, tự sáng tạo ra nội dung mà không có trong tài liệu gốc.
## LƯU Ý ĐẶC BIỆT
- Trang tài liệu có thể bắt đầu ở giữa câu hỏi hoặc đáp án (do cắt từ tài liệu lớn) nên không được bỏ qua trích xuất nội dung dù có vẻ không hoàn chỉnh
- Định dạng AssessmentMarkupLanguage bạn đang xuất ra không phải Latex, cũng không phải HTML nên không được dùng nhiều hơn các tính năng, quy định có sẵn
```

---
File: run_worker.py
---
```python
import asyncio
from app.worker.instance import worker
from app.services.logging_config import get_logger

logger = get_logger()



# --- IMPORT TASKS ---
import app.worker.tasks.process_pdf
import app.worker.tasks.process_img
import app.worker.tasks.extractor
import app.worker.tasks.document_parser

async def main():
    logger.info("Starting worker...")
    await worker.run_worker()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Worker stopped by user.")
    except Exception as e:
        logger.exception(f"FAILED to process : {e}")
```

---
File: services/__init__.py
---
```python

```

---
File: services/__pycache__/__init__.cpython-310.pyc
---
Error reading file: 'utf-8' codec can't decode byte 0x91 in position 10: invalid start byte

---
File: services/__pycache__/counter.cpython-310.pyc
---
Error reading file: 'utf-8' codec can't decode byte 0xe5 in position 8: invalid continuation byte

---
File: services/__pycache__/create_task.cpython-310.pyc
---
Error reading file: 'utf-8' codec can't decode byte 0x91 in position 10: invalid start byte

---
File: services/__pycache__/logging_config.cpython-310.pyc
---
Error reading file: 'utf-8' codec can't decode bytes in position 9-10: invalid continuation byte

---
File: services/counter.py
---
```python
from app.lib.redis_client import redis_manager

r = redis_manager.get_connection()
async def increment_counter(id : str):
    return await r.incr(f"counter:{id}")

async def get_counter(id : str):
    value = await r.get(f"counter:{id}")
    return int(value) if value else 0

async def decrement_counter(id : str):
    return await r.decr(f"counter:{id}")

async def set_counter(id : str, value : int):
    return await r.set(f"counter:{id}", value=value)

async def delete_counter(id : str):
    await r.delete(f"counter:{id}")
```

---
File: services/create_task.py
---
```python
# services/create_task.py
from datetime import datetime, timezone
from typing import List
import uuid
from sqlalchemy.orm import Session
from app.db.models import Task, TaskStatus, TaskType
from app.db.models import InferenceLog

def create_task(task_type: TaskType, session: Session) -> Task:
    new_task = Task(
        type=task_type,
        status=TaskStatus.pending,
    )
    session.add(new_task)
    session.commit()
    session.refresh(new_task)
    return new_task

def create_log(imageUrls : str,
            objectKeys : List[str],
            objectUrls : List[str],
            requestId : str,
            num_input_token : int,
            num_output_token : int,
            rawOutput : str,
            page_order : int,
            session : Session):
    new_log = InferenceLog(
        imageUrls=imageUrls,
        objectKeys=objectKeys,
        objectUrls=objectUrls,
        requestId=requestId,
        num_input_token=num_input_token,
        num_output_token=num_output_token,
        rawOutput=rawOutput,
        page_order=page_order,
    )
    session.add(new_log)
    session.commit()
    session.refresh(new_log)
    return new_log

def get_task_by_id(session : Session, task_id: str):
    return session.query(Task).filter(Task.id == task_id).first()

def get_logs_by_task(session: Session, task_id: str) -> List[InferenceLog]:
    return session.query(InferenceLog).filter(InferenceLog.requestId == task_id).all()

def update_task_result(session : Session, task_id: str, result : str):
    task = session.query(Task).filter(Task.id == task_id).first()
    task.result = result
    task.updatedAt = datetime.now(timezone.utc)
    task.status = TaskStatus.done
```

---
File: services/logging_config.py
---
```python
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
```

---
File: utils/__init__.py
---
```python

```

---
File: utils/__pycache__/__init__.cpython-310.pyc
---
Error reading file: 'utf-8' codec can't decode byte 0xc9 in position 8: invalid continuation byte

---
File: utils/__pycache__/agent.cpython-310.pyc
---
Error reading file: 'utf-8' codec can't decode byte 0xca in position 8: invalid continuation byte

---
File: utils/__pycache__/annotate.cpython-310.pyc
---
Error reading file: 'utf-8' codec can't decode byte 0xec in position 8: invalid continuation byte

---
File: utils/__pycache__/draw_boxes.cpython-310.pyc
---
Error reading file: 'utf-8' codec can't decode byte 0x91 in position 10: invalid start byte

---
File: utils/__pycache__/process_and_annotate.cpython-310.pyc
---
Error reading file: 'utf-8' codec can't decode bytes in position 9-10: invalid continuation byte

---
File: utils/__pycache__/remove_padding.cpython-310.pyc
---
Error reading file: 'utf-8' codec can't decode byte 0x91 in position 10: invalid start byte

---
File: utils/__pycache__/upload_r2.cpython-310.pyc
---
Error reading file: 'utf-8' codec can't decode byte 0x91 in position 10: invalid start byte

---
File: utils/agent.py
---
```python
import asyncio
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
from typing import Union
from google.genai import errors
from app.services.logging_config import get_logger
import redis.asyncio as redis
import time
# Load biến môi trường và lấy danh sách API keys
load_dotenv()
logger = get_logger()
keys = os.environ.get("GEMINI_API_KEY", "").split(",")

from app.lib.redis_client import redis_manager
r = redis_manager.get_connection()
async def init_api_key_zset(keys):
    logger.info("Successed in initializing api keys set")
    # chỉ chạy một lần lúc khởi động
    now = 0.0
    # mapping API key -> score
    mapping = { key: now for key in keys }
    # nếu cần đảm bảo ZSET trống
    await r.delete("gemini_api_keys")
    # thêm tất cả keys với score 0
    await r.zadd("gemini_api_keys", mapping)

# Logger


async def get_lru_api_key():
    while True:
        result = await r.zrange("gemini_api_keys", 0, 0, withscores=True)
        if not result:
            raise RuntimeError("None of api keys are presence in the API keys set")

        key, _old_score = result[0]

        key = key.decode()
        idx = keys.index(key)
        logger.info(f"Key {idx} has been selected")
        now_ts = time.time()
        updated = await r.zadd(
            "gemini_api_keys",
            {key: now_ts},
            xx=True,
            ch=True
        )
        if updated:
            return key

        await asyncio.sleep(0)


async def GeminiAgent(
    model: str,
    contents: Union[types.ContentListUnion, types.ContentListUnionDict],
    config: types.GenerateContentConfigOrDict,
    retry_delay: float = 1.0,
    max_retries: int = 2
):
    delay = retry_delay
    retry_count = 0

    while True:
        # Lấy key LRU
        api_key = await get_lru_api_key()
        client = genai.Client(api_key=api_key)

        try:
            response = await client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )

            if not isinstance(response.text, str) or not response.text.strip():
                raise ValueError("GeminiAgent: response.text is không hợp lệ")

            return response

        except errors.APIError as e:
            code = e.code
            msg = e.message or ""
            logger.warning(f"[GeminiAgent] APIError (Code {code}): {msg}")

            if code == 429:
                # Chỉ cần log và loop tiếp để lấy key khác
                logger.info(f"[GeminiAgent] Key `{api_key}` bị rate-limit, rotate sang key khác.")
                # reset delay cho lần dùng key mới
                delay = retry_delay
                continue

            if str(code).startswith("5"):
                logger.info(f"[GeminiAgent] Lỗi server {code}, chờ {delay:.1f}s rồi thử lại...")
                await asyncio.sleep(delay)
                delay = min(delay * 1.5, 60)
                continue

            # Các lỗi 4xx khác – retry tối đa
            retry_count += 1
            if retry_count > max_retries:
                logger.error(f"[GeminiAgent] Lỗi client {code} vượt quá {max_retries} lần, dừng.")
                raise
            logger.warning(f"[GeminiAgent] Lỗi client {code}, retry {retry_count}/{max_retries} sau {delay:.1f}s...")
            await asyncio.sleep(delay)
            delay = min(delay * 1.5, 60)

        except Exception as e:
            logger.error(f"[GeminiAgent] Lỗi không xác định: {type(e).__name__} – {e}")
            raise

if __name__ == "__main__":
    async def main():
        await init_api_key_zset(keys)

        count = 0
        start_time = time.time()

        while True:
            try:
                response = await GeminiAgent(
                    model="gemini-2.0-flash",
                    contents=[
                        types.Content(
                            role="user",
                            parts=[types.Part.from_text(text="xin chào")]
                        )
                    ],
                    config=types.GenerateContentConfig()
                )
                print(f"Phản hồi: {response.text.strip()}")
                count += 1

            except Exception as e:
                logger.error(f"[main] Lỗi khi gửi request: {str(e)}")

            # In thống kê mỗi phút
            if time.time() - start_time >= 60:
                print(f"Số lần gọi API thành công trong 1 phút: {count}")
                count = 0
                start_time = time.time()

            await asyncio.sleep(1)  # Gửi request mỗi giây

    asyncio.run(main())
```

---
File: utils/draw_boxes.py
---
```python
import cv2
import requests
import numpy as np
from PIL import Image
from typing import Tuple, Optional, List
from io import BytesIO 
import concurrent.futures
from functools import lru_cache
import base64
from dotenv import load_dotenv
load_dotenv()
import os
from collections import OrderedDict
# Load model

# --------- CACHE CHO COLORS VÀ CONSTANTS ----------
COLORS = [(0,255,0), (255,0,0), (0,0,255), (0,255,255), (255,0,255), (255,255,0)]
FONT = cv2.FONT_HERSHEY_SIMPLEX

@lru_cache(maxsize=None)
def get_text_size_cached(text: str, font_scale: float, thickness: int):
    return cv2.getTextSize(text, FONT, font_scale, thickness)


def calculate_max_font_scale(text: str, max_width: int, max_height: int,
                             font_face=cv2.FONT_HERSHEY_SIMPLEX,
                             initial_scale: float = 1.0,
                             scale_step: float = 0.5,
                             min_scale_step: float = 0.01,
                             padding: int = 2) -> Tuple[float, int]:
    def fits(s: float) -> bool:
        th = max(1, int(s * 2))
        (w_text, h_text), baseline = cv2.getTextSize(text, font_face, s, th)
        total_h = h_text + baseline
        return (w_text + 2 * padding) <= max_width and (total_h + 2 * padding) <= max_height

    scale = initial_scale
    if not fits(scale):
        while scale > min_scale_step and not fits(scale):
            scale /= 2
    else:
        while fits(scale):
            scale += scale_step
        scale -= scale_step

    step = scale_step
    while step >= min_scale_step:
        if fits(scale + step):
            scale += step
        else:
            step /= 2

    thickness = max(1, int(scale * 2))
    return scale, thickness


def draw_boxes(
    img: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    boxes: np.ndarray,
    class_names: Optional[List[str]] = None,
    base_box_thickness: int = 2,
    skip_class_ids: Optional[List[int]] = None,
    skip_class_names: Optional[List[str]] = None,
    draw_labels: bool = True,
    sort_by_coordinate: bool = True,
    row_threshold: int = 10
):
    if boxes is None or len(boxes) == 0:
        return img.copy(), {}

    h_img, w_img = img.shape[:2]
    box_thickness = max(1, int(base_box_thickness * min(w_img, h_img) / 1000))
    out = img.copy()
    skip_ids = set(skip_class_ids or [])
    skip_names = set(skip_class_names or [])

    bboxes = boxes.astype(np.int32)
    valid_indices = []
    for i, bbox in enumerate(bboxes):
        cls = int(labels[i])
        if cls in skip_ids:
            continue
        name = (class_names[cls] if class_names and cls < len(class_names) else str(cls))
        if name in skip_names:
            continue
        x1, y1, x2, y2 = bbox
        if (x2 - x1) < 10 or (y2 - y1) < 10:
            continue
        valid_indices.append(i)

    # Sort theo y-then-x
    if sort_by_coordinate and valid_indices:
        entries = [(i, *bboxes[i][:4]) for i in valid_indices]
        entries.sort(key=lambda e: e[2])  # sort theo y1
        rows = []
        for i, x1, y1, x2, y2 in entries:
            placed = False
            for row in rows:
                y_min, y_max, lst = row
                if y1 <= y_max + row_threshold and y2 >= y_min - row_threshold:
                    row[0] = min(y_min, y1);
                    row[1] = max(y_max, y2)
                    lst.append((i, x1))
                    placed = True
                    break
            if not placed:
                rows.append([y1, y2, [(i, x1)]])
        rows.sort(key=lambda r: r[0])
        sorted_indices = []
        for _, _, group in rows:
            group.sort(key=lambda e: e[1])
            sorted_indices += [i for i, _ in group]
    else:
        sorted_indices = valid_indices

    # Tạo dict cho ROI
    cropped_objects_np = OrderedDict()

    for order_idx, i in enumerate(sorted_indices):
        x1, y1, x2, y2 = bboxes[i]
        cls = int(labels[i])
        name = (class_names[cls] if class_names and cls < len(class_names) else str(cls))
        prefix = name[:2].upper() if len(name) >= 2 else name.upper()
        key = f"{prefix}{order_idx + 1}"

        # Vẽ khung
        color = COLORS[cls % len(COLORS)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, box_thickness)

        if draw_labels:
            font_scale, font_thickness = calculate_max_font_scale(
                key, x2-x1, y2-y1, font_face=FONT
            )
            (w_text, h_text), baseline = get_text_size_cached(key, font_scale, font_thickness)
            text_x = np.clip(x1 + (x2-x1 - w_text)//2, x1+2, x2 - w_text - 2)
            text_y = np.clip(y1 + h_text + 2, y1 + h_text + 2, y2 - baseline - 2)

            # ========== Layer 1: NỀN ==========
            background_overlay = out.copy()
            cv2.rectangle(background_overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
            alpha_background = 0.4  # nền mờ nhẹ
            cv2.addWeighted(background_overlay, alpha_background, out, 1 - alpha_background, 0, out)

            # ========== Layer 2: CHỮ ==========
            text_overlay = out.copy()
            cv2.putText(text_overlay, key, (text_x, text_y), FONT, font_scale, (0, 0, 255), font_thickness)
            alpha_text = 0.75  # chữ gần như rõ ràng
            cv2.addWeighted(text_overlay, alpha_text, out, 1 - alpha_text, 0, out)



        roi = img[y1:y2, x1:x2]
        if roi.size > 0:
            cropped_objects_np[key] = roi.copy()

    return out, cropped_objects_np
```

---
File: utils/remove_padding.py
---
```python
import cv2
import numpy as np

def remove_white_padding(image_np: np.ndarray) -> np.ndarray:
    """
    Loại bỏ padding trắng xung quanh nội dung chính của ảnh.
    Nhận và trả về ảnh dưới dạng NumPy array (BGR).
    
    Args:
        image_np (np.ndarray): Ảnh đầu vào dưới dạng NumPy (BGR).
        
    Returns:
        np.ndarray: Ảnh đã được crop bỏ padding trắng.
    """
    if image_np is None or image_np.size == 0:
        return image_np

    # 1. Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    # 2. Invert the image: white → 0, text/dark → 255
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # 3. Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image_np  # No content found

    # 4. Find bounding box around all contours
    all_contours = np.vstack(contours)
    x, y, w, h = cv2.boundingRect(all_contours)

    # 5. Crop and return
    cropped = image_np[y:y+h, x:x+w]
    return cropped
```

---
File: utils/upload_r2.py
---
```python
# upload_r2.py

import os
import base64
import asyncio
import aioboto3
from urllib.parse import urlparse
from typing import List
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
# Load environment variables from .env
load_dotenv()

# Required ENV variables
ENDPOINT_URL_R2 = os.getenv("ENDPOINT_URL_R2")
AWS_ACCESS_KEY_ID_R2 = os.getenv("AWS_ACCESS_KEY_ID_R2")
AWS_SECRET_ACCESS_KEY_R2 = os.getenv("AWS_SECRET_ACCESS_KEY_R2")
PUBLIC_URL_R2 = os.getenv("PUBLIC_URL_R2")

if not all([ENDPOINT_URL_R2, AWS_ACCESS_KEY_ID_R2, AWS_SECRET_ACCESS_KEY_R2, PUBLIC_URL_R2]):
    raise RuntimeError("Missing required environment variables for R2")

# Parse bucket and endpoint
parsed = urlparse(ENDPOINT_URL_R2)
bucket_name = parsed.path.lstrip("/")  # remove leading slash
endpoint_base = f"{parsed.scheme}://{parsed.netloc}"

# === FNV-1a hash (simple 32-bit) ===
def fnv1a_hash(buffer: bytes) -> str:
    hash_ = 0x811c9dc5
    for byte in buffer:
        hash_ ^= byte
        hash_ = (hash_ * 0x01000193) % (1 << 32)
    return format(hash_, "08x")

# === Upload single image (base64 JPEG) ===
async def upload_img_to_r2(buffer: bytes) -> str:
    with Image.open(BytesIO(buffer)) as img:
        output = BytesIO()
        img.convert("RGB").save(output, format="JPEG", quality=85, optimize=True)
        jpeg_bytes = output.getvalue()

    key = fnv1a_hash(jpeg_bytes) + ".jpg"

    session = aioboto3.Session()
    async with session.client(
        "s3",
        endpoint_url=endpoint_base,
        aws_access_key_id=AWS_ACCESS_KEY_ID_R2,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY_R2,
        region_name="auto",
    ) as s3:
        await s3.put_object(
            Bucket=bucket_name,
            Key=key,
            Body=jpeg_bytes,
            ContentType="image/jpeg",
            ACL="public-read"
        )

    return f"{PUBLIC_URL_R2}/{key}"
# === Upload multiple images with concurrency limit ===
async def upload_multiple_images(buffers: List[bytes], concurrency_limit: int = 5) -> List[str]:
    sem = asyncio.Semaphore(concurrency_limit)

    async def limited_upload(buffer: bytes):
        async with sem:
            return await upload_img_to_r2(buffer)

    tasks = [limited_upload(buf) for buf in buffers]
    return await asyncio.gather(*tasks)
```

---
File: worker/__init__.py
---
```python

```

---
File: worker/__pycache__/__init__.cpython-310.pyc
---
Error reading file: 'utf-8' codec can't decode byte 0x91 in position 10: invalid start byte

---
File: worker/__pycache__/core.cpython-310.pyc
---
Error reading file: 'utf-8' codec can't decode byte 0xc0 in position 8: invalid start byte

---
File: worker/__pycache__/extractor_worker.cpython-310.pyc
---
Error reading file: 'utf-8' codec can't decode byte 0xfa in position 8: invalid start byte

---
File: worker/__pycache__/instance.cpython-310.pyc
---
Error reading file: 'utf-8' codec can't decode byte 0x91 in position 10: invalid start byte

---
File: worker/__pycache__/process_img.cpython-310.pyc
---
Error reading file: 'utf-8' codec can't decode byte 0xbf in position 8: invalid start byte

---
File: worker/__pycache__/process_pdf.cpython-310.pyc
---
Error reading file: 'utf-8' codec can't decode byte 0xfb in position 8: invalid start byte

---
File: worker/instance.py
---
```python
from app.lib.worker_core import SimpleRedisWorker

worker = SimpleRedisWorker()
```

---
File: worker/tasks/__init__.py
---
```python

```

---
File: worker/tasks/__pycache__/__init__.cpython-310.pyc
---
Error reading file: 'utf-8' codec can't decode byte 0x91 in position 10: invalid start byte

---
File: worker/tasks/__pycache__/document_parser.cpython-310.pyc
---
Error reading file: 'utf-8' codec can't decode byte 0x9d in position 8: invalid start byte

---
File: worker/tasks/__pycache__/extractor.cpython-310.pyc
---
Error reading file: 'utf-8' codec can't decode byte 0xb0 in position 8: invalid start byte

---
File: worker/tasks/__pycache__/process_img.cpython-310.pyc
---
Error reading file: 'utf-8' codec can't decode byte 0x9f in position 8: invalid start byte

---
File: worker/tasks/__pycache__/process_pdf.cpython-310.pyc
---
Error reading file: 'utf-8' codec can't decode byte 0xdb in position 8: invalid continuation byte

---
File: worker/tasks/document_parser.py
---
```python
import asyncio
import logging

from google.genai import types
import re
from app.worker.instance import worker
from app.utils.agent import GeminiAgent
from app.prompt import (
    get_initial_slurp_prompt,
    get_slurp_continuation_prompt,
    get_initial_slurp_examples,
    get_slurp_continuation_examples
)
import json  
from app.postprocessing.replaceimgfig import replace_img_to_fig, replace_fig2img_immutable
from app.db.client import get_session
from app.db.models import TaskType
from app.services.create_task import create_task, update_task_result
from app.postprocessing.slurp2json import slurp_to_json, autofix_missing_pipes
from app.services.logging_config import get_logger

logger = get_logger()

PARSER_MODEL = "gemini-2.5-flash"
MAX_CONTINUATION_ATTEMPTS = 5


def clean_output(text: str) -> str:
    # Loại bỏ tất cả marker @slurp_...@
    return re.sub(r"@slurp(_resume)?(_incomplete)?_(start|end)@", "", text).strip()

async def llmAsParser(text: str):
    text, figures = replace_img_to_fig(text)

    initial_contents = get_initial_slurp_examples() + [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=f"@mml_start@{text}@mml_end@")]
        )
    ]
    initial_instruction = get_initial_slurp_prompt()
    initial_config = types.GenerateContentConfig(
        system_instruction=[types.Part.from_text(text=initial_instruction)],
        temperature=0.2
    )

    initial_response = await GeminiAgent(
        model=PARSER_MODEL,
        contents=initial_contents,
        config=initial_config
    )
    slurp_content = initial_response.text

    if "@slurp_start@" in slurp_content and "@slurp_end@" in slurp_content:
        return clean_output(slurp_content), figures

    if "@slurp_start@" not in slurp_content:
        raise ValueError("Parser did not return start marker.")

    slurp_content = clean_output(slurp_content)
    cont_instruction = get_slurp_continuation_prompt()
    continuation_config = types.GenerateContentConfig(
        system_instruction=[types.Part.from_text(text=cont_instruction)],
        temperature=0.2
    )
    continuation_examples = get_slurp_continuation_examples()

    for attempt in range(MAX_CONTINUATION_ATTEMPTS):
        cont_contents = continuation_examples + [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=f"@mml_start@{text}@mml_end@\n"
                             f"@slurp_incomplete_start@{slurp_content}@slurp_incomplete_end@"
                    )
                ]
            )
        ]

        cont_response = await GeminiAgent(
            model=PARSER_MODEL,
            contents=cont_contents,
            config=continuation_config,
        )
        cont_output = cont_response.text

        if "@slurp_resume_end@" in cont_output or "@slurp_end@" in cont_output:
            slurp_content += clean_output(cont_output)
            return slurp_content, figures

        slurp_content += clean_output(cont_output)

    logger.info("Parser failed to complete after maximum retries.")
    return slurp_content, figures

@worker.task(name="documentParsing", max_concurrency=20, max_retries=0)
async def parse_document(task_id: str, text: str):
    logger.info(f"Parsing document {task_id}")
    try:
        slurp_content, figures = await llmAsParser(text=text)
        fixed_slurp_content = autofix_missing_pipes(slurp_content)
        parsed_json = slurp_to_json(fixed_slurp_content)
        
        refine_json = replace_fig2img_immutable(parsed_json, figures)
        
        dumped_json = json.dumps({"raw" : slurp_content, "parsed": refine_json }, ensure_ascii=False)
        
        with get_session() as session:
            update_task_result(session=session, task_id=task_id, result=dumped_json)  
        logger.info(f"[{task_id}] Task completed successfully.")
    except Exception as e:
        logger.exception(f"[{task_id}] Failed to parse document : {e}")


if __name__ == "__main__":
    async def main():
        from pathlib import Path
        import json

        # 1. Đọc input
        input_file = Path("tests/de15.txt")
        text = input_file.read_text(encoding="utf-8")

        # 2. Chạy parser
        slurp_content, figures = await llmAsParser(text)
        # 3. Chuyển thành JSON và thay fig → img
        parsed_json = slurp_to_json(autofix_missing_pipes(slurp_content))
        refine_json = replace_fig2img_immutable(parsed_json, figures)

        # 4. Ghi kết quả cuối cùng (có URL) ra file
        output_path = Path("tests/output.json")
        output_path.write_text(
            json.dumps(refine_json, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"✅ Đã ghi kết quả có URL vào {output_path}")

    asyncio.run(main())
```

---
File: worker/tasks/extractor.py
---
```python
# worker/tasks/extractor.py
from typing import List
from app.prompt import get_extraction_figure_prompt, get_extraction_non_figure_prompt
from app.worker.instance import worker
from app.utils.agent import GeminiAgent
from google.genai import types
import asyncio
from app.main import get_file_bytes
from PIL import Image
from io import BytesIO
from app.utils.upload_r2 import upload_multiple_images
from app.postprocessing.raw_response import extract_response, replace_image_tags
import base64
from app.services.create_task import create_log, get_logs_by_task, update_task_result, get_task_by_id
from app.db.client import get_session
from app.services.counter import decrement_counter, delete_counter
from app.db.models import InferenceLog, TaskStatus, Task # THÊM MỚI
from app.services.logging_config import get_logger

logger = get_logger()


async def uploadImageFromUrls(ImageUrls) -> List[str]:
    if not ImageUrls:
        return []
    image_bytes = await asyncio.gather(*(get_file_bytes(img) for img in ImageUrls))
    image_urls = await upload_multiple_images(image_bytes, concurrency_limit=10)
    return image_urls


async def convert_to_webp_base64(img_bytes: bytes, quality: int = 80) -> str:
    with Image.open(BytesIO(img_bytes)) as img:
        output = BytesIO()
        img.convert("RGBA").save(output, format="WEBP", quality=quality, method=6)
        webp_bytes = output.getvalue()
        base64_str = base64.b64encode(webp_bytes).decode('utf-8')
        return base64_str

@worker.task(name="parseDocumentImage", max_concurrency=20, max_retries=0)
async def extractDocumentImage(task_id: str, img_url: str, page_order: int, cropped_objects_urls: List[tuple]):
    try:
        logger.info(f"Start processing task {task_id}, page {page_order}")

        img_bytes = await get_file_bytes(img_url)
        img_webp = await convert_to_webp_base64(img_bytes)

        object_urls = [url for key, url in cropped_objects_urls]
        object_keys = [key for key, url in cropped_objects_urls]

        user_parts = [
            types.Part.from_bytes(
                mime_type="image/webp",
                data=base64.b64decode(img_webp),
            ),
        ]

        llmResponse = None
        uploadedUrls = []

        if object_urls:
            model = "gemini-2.0-flash-001"
            generate_content_config = types.GenerateContentConfig(
                media_resolution="MEDIA_RESOLUTION_HIGH",
                system_instruction=[
                    types.Part.from_text(text=get_extraction_figure_prompt()),
                ],
                temperature=1
            )
            contents = [types.Content(role="user", parts=user_parts)]
            
            llmResponse, uploadedUrls = await asyncio.gather(
                GeminiAgent(model=model, contents=contents, config=generate_content_config),
                uploadImageFromUrls(object_urls)
            )
        else:
            generate_content_config = types.GenerateContentConfig(
                media_resolution="MEDIA_RESOLUTION_HIGH",
                system_instruction=[
                    types.Part.from_text(text=get_extraction_non_figure_prompt()),
                ],
                temperature=1
            )
            contents = [types.Content(role="user", parts=user_parts)]
            llmResponse = await GeminiAgent(model="gemini-2.0-flash-001", contents=contents, config=generate_content_config)

        if not llmResponse or not llmResponse.text:
            raise ValueError("LLM response is invalid or empty!")

        with get_session() as session:
            create_log(
                imageUrls=img_url,
                objectKeys=object_keys,
                objectUrls=uploadedUrls,
                requestId=task_id,
                num_input_token=llmResponse.usage_metadata.prompt_token_count or 0,
                num_output_token=llmResponse.usage_metadata.candidates_token_count or 0,
                rawOutput=llmResponse.text,
                page_order=page_order,
                session=session
            )
        
        # SỬA LỖI: Logic counter mới, đảm bảo tính nguyên tử và chính xác
        remaining_pages = await decrement_counter(task_id)
        logger.info(f"[{task_id}] page {page_order} processed. Remaining pages to process: {remaining_pages}")

        # Nếu không còn trang nào cần xử lý (counter <= 0), tiến hành tổng hợp kết quả
        if remaining_pages <= 0:
                logger.info(f"[{task_id}] All pages processed or last page. Attempting to aggregate results.")
                await delete_counter(task_id)

                with get_session() as session:
                    task = session.query(Task).filter(Task.id == task_id).with_for_update().first()
                    if task.status != TaskStatus.pending: 
                        logger.warning(f"[{task_id}] Task is no longer in 'pending' state (current: {task.status.value}). Skipping aggregation.")
                        return 

                    combined_output = ''
                    logs: list[InferenceLog] = get_logs_by_task(session=session, task_id=task_id)
                    sorted_logs = sorted(logs, key=lambda log: log.page_order)

                    for log in sorted_logs:
                        doc_block = extract_response(log.rawOutput).document
                        if log.objectUrls and doc_block:
                            image_map = dict(zip(list(log.objectKeys), list(log.objectUrls)))
                            doc_block, _ = replace_image_tags(doc_block, image_map)
                        
                        if doc_block:
                            combined_output += doc_block.strip() + "\n"
                    
                    update_task_result(session=session, result=combined_output.strip(), task_id=task_id)
                    logger.info(f"[{task_id}] Successfully wrote final result for task ")

    except Exception as e:
        logger.exception(f"[{task_id}] FAILED to process task, page {page_order}: {e}")
        with get_session() as session:
            task = get_task_by_id(session=session, task_id=task_id)
            if task and task.status == TaskStatus.pending:
                task.status = TaskStatus.failed
                task.error = f"Error on page {page_order}: {str(e)}"
                session.commit()
        await delete_counter(task_id)
```

---
File: worker/tasks/process_img.py
---
```python
# worker/extractor_worker.py
import asyncio
import os
from app.db.client import get_session
from app.db.models import Task, TaskStatus
import fitz
import numpy as np
import requests
import hashlib
from app.main import save_file, get_file_bytes
from pathlib import Path
import io
import base64
import cv2
from io import BytesIO
from PIL import Image
from app.ml_models.image_classifier import classifier
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from app.utils.remove_padding import remove_white_padding
from app.utils.draw_boxes import draw_boxes
from app.ml_models.rfdetr import rtdetr_model
from app.utils.upload_r2 import upload_multiple_images # <-- IMPORT MỚI
from collections import OrderedDict # <-- IMPORT MỚI
from app.worker.instance import worker

import io
import cv2
import numpy as np
from typing import Dict, Tuple
from pathlib import Path
from PIL import Image
import asyncio

JPEG_QUALITY = 90
PIL_JPEG_PARAMS = {'format': 'JPEG', 'quality': JPEG_QUALITY}
JPEG_PARAMS = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
from app.services.logging_config import get_logger

logger = get_logger()

def get_annotated_images(image: np.ndarray,  threshold: float = 0.5):
    cropped_image = remove_white_padding(image)
    pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    scores, labels, boxes = rtdetr_model.run_inference(pil_image, confidence_threshold=threshold)
    
    # draw_boxes giờ sẽ trả về `OrderedDict` của các numpy array
    out_np, cropped_objects_np = draw_boxes(
        cropped_image.copy(), scores, labels, boxes,
        class_names=["Image", "Table"],
        skip_class_ids=[1], skip_class_names=["Table"],
        draw_labels=True
    )
    return out_np, cropped_objects_np


BASE_URL = "http://localhost:8000/media/"

async def save_cropped_objects_to_urls(
    cropped_objects_np: Dict[str, np.ndarray],
    image_format: str = "jpg"
) -> Dict[str, str]:


    async def encode_and_save(key: str, img: np.ndarray) -> Tuple[str, str]:
        # Encode np.ndarray -> bytes (JPG)
        is_success, buffer = cv2.imencode(f".{image_format}", img)
        if not is_success:
            raise ValueError(f"Failed to encode image for key: {key}")
        img_bytes = buffer.tobytes()

        # Gọi hàm lưu file
        relative_path: Path = await save_file(img_bytes, image_format)

        # Trả lại URL đầy đủ
        return key, f"{BASE_URL}{relative_path.as_posix()}"

    # Chạy đồng thời tất cả encode + save
    tasks = [encode_and_save(key, img) for key, img in cropped_objects_np.items()]
    results = await asyncio.gather(*tasks)

    return dict(results)


# Hàm này cần là async vì nó sẽ gọi upload_multiple_images
async def annotate_img(img_np: np.ndarray):
    pred_idx, logits, confidence = classifier.predict(img_np)

    # Khởi tạo để tránh lỗi nếu không phải loại cần xử lý
    processed_img_np = img_np.copy()
    cropped_objects_np = {}

    if pred_idx == 1:
        processed_img_np, cropped_objects_np = get_annotated_images(img_np, threshold=0.5)

    # Lưu các object được crop
    cropped_objects_urls = await save_cropped_objects_to_urls(cropped_objects_np)

    is_success, buffer = cv2.imencode(".jpg", processed_img_np)
    if not is_success:
        raise ValueError("Failed to encode processed image")

    processed_img_bytes = buffer.tobytes()
    processed_relative_path: Path = await save_file(processed_img_bytes, "jpg")
    processed_img_url = f"{BASE_URL}{processed_relative_path.as_posix()}"

    return processed_img_url, cropped_objects_urls




@worker.task(name="process_img",max_concurrency=1, max_retries=1)
async def process_img(task_id :str,page_idx: int, file_url: str):
    logger.info(f"{task_id} Start processing task ")
    try:
        img_bytes = await get_file_bytes(file_url)
        img = Image.open(BytesIO(img_bytes)).convert('RGB') 
        img_np_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        processed_img_url , cropped_objects_urls = await annotate_img(img_np=img_np_bgr)
        
        await worker.enqueue("parseDocumentImage", task_id , processed_img_url, page_idx, list(cropped_objects_urls.items()))
    except Exception as e:
        logger.exception(f"[{task_id}] FAILED to process page {page_idx}: {e}")
    # Tại đây, cropped_objects_urls đã có cấu trúc Key-URL như mong muốn
    # TODO: Cập nhật task trong DB với thông tin cropped_objects_urls
    # Ví dụ (cần import json nếu lưu vào Text field, hoặc có thể lưu dưới dạng JSONB nếu DB hỗ trợ):
    # import json
    # from app.db.client import get_session
    # from app.db.models import Task, TaskStatus

    # try:
    #     with get_session() as session:
    #         task = session.query(Task).filter(Task.id == ctx['task_id']).first() # Giả định task_id được truyền qua ctx
    #         if task:
    #             # Kết quả có thể bao gồm cả ảnh gốc đã annotate (nếu muốn) và các URL của cropped_objects
    #             task.result = json.dumps({
    #                 "annotated_image_b64": img_b64, # Hoặc upload cái này lên R2 và lưu URL
    #                 "cropped_objects": cropped_objects_urls
    #             })
    #             task.status = TaskStatus.done
    #             session.commit()
    #             print(f"[Worker] Task {ctx['task_id']} marked as done with cropped objects URLs.")
    #         else:
    #             print(f"[Worker] Task {ctx['task_id']} not found.")
    # except Exception as e:
    #     print(f"[Worker] Error processing task {ctx['task_id']}: {e}")
    #     with get_session() as session:
    #         task = session.query(Task).filter(Task.id == ctx['task_id']).first()
    #         if task:
    #             task.status = TaskStatus.failed
    #             task.error = str(e)
    #             session.commit()
```

---
File: worker/tasks/process_pdf.py
---
```python
# worker/tasks/process_pdf.py

import fitz
from app.main import save_file, get_file_bytes
from app.worker.instance import worker
from app.services.counter import set_counter
from app.services.logging_config import get_logger
# THÊM MỚI: Import để xử lý trường hợp 0 trang
from app.db.client import get_session
from app.services.create_task import update_task_result

logger = get_logger()

@worker.task(name="process_pdf", max_concurrency=1, max_retries=1)
async def process_pdf(task_id: str, file_url: str):
    logger.info(f"Start processing task {task_id}")
    try:
        pdf_bytes = await get_file_bytes(file_url)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        num_pages = len(doc)
        
        # CẢI TIẾN: Khởi tạo counter bằng tổng số trang cho trực quan.
        await set_counter(task_id, num_pages)

        # CẢI TIẾN: Xử lý trường hợp PDF không có trang nào.
        if num_pages == 0:
            logger.warning(f"{task_id} PDF for task has 0 pages. Finishing task immediately.")
            with get_session() as session:
                update_task_result(session=session, result="", task_id=task_id)
            return

        for page_idx in range(num_pages):
            page = doc[page_idx]
            pix = page.get_pixmap(dpi=200, alpha=False)
            img_bytes = pix.tobytes("jpeg")
            filepath = await save_file(file_bytes=img_bytes, file_type='jpg')
            file_url = f"http://localhost:8000/media/{filepath}"
            await worker.enqueue("process_img", task_id, page_idx, file_url)

        logger.info(f"[{task_id}] Enqueued {num_pages} pages for task ")
    except Exception as e:

        logger.exception(f"[{task_id}] Error processing task {task_id}: {e}")
    # try:
    #     with get_session() as session:
    #         task = session.query(Task).filter(Task.id == task_id).first()
    #         if task:
    #             task.status = TaskStatus.done
    #             task.result = "mock result"
    #             session.commit()
    #             print(f"[Worker] Task {task_id} marked as done")
    #         else:
    #             print(f"[Worker] Task {task_id} not found")
    # except Exception as e:
    #     with get_session() as session:
    #         task = session.query(Task).filter(Task.id == task_id).first()
    #         if task:
    #             task.status = TaskStatus.failed
    #             task.error = str(e)
    #             session.commit()
    #     print(f"[Worker] Error processing task {task_id}: {e}")
```

