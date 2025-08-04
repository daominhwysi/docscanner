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
        task = create_task(task_type=TaskType("parseDocumentPDF"), session=session)
        task_id = task.id

    await worker.enqueue("process_pdf", task_id, file_url)

    return {"task_id": task_id}


class Text2Slurp(BaseModel):
    text: str

@app.post("/document-parsing")
async def documentParsing(body: Text2Slurp):
    task_id=None
    with get_session() as session:
        task = create_task(task_type=TaskType("documentParsing"), session=session)
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
