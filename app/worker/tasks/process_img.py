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
    logger.info(f"[Worker-Image] Start processing task {task_id}")
    img_bytes = await get_file_bytes(file_url)
    img = Image.open(BytesIO(img_bytes))
    
    # Gọi hàm annotate_img (giờ đã là async)
    processed_img_url , cropped_objects_urls = await annotate_img(img_np=np.array(img))

    await worker.enqueue("parseDocumentImage", task_id , processed_img_url, page_idx, list(cropped_objects_urls.items()))

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