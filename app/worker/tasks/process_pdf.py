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
    logger.info(f"[Worker-PDF] Start processing task {task_id}")
    pdf_bytes = await get_file_bytes(file_url)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    num_pages = len(doc)
    
    # CẢI TIẾN: Khởi tạo counter bằng tổng số trang cho trực quan.
    await set_counter(task_id, num_pages)

    # CẢI TIẾN: Xử lý trường hợp PDF không có trang nào.
    if num_pages == 0:
        logger.warning(f"PDF for task {task_id} has 0 pages. Finishing task immediately.")
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

    logger.info(f"[Worker-PDF] Enqueued {num_pages} pages for task {task_id}")
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
