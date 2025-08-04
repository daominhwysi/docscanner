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
from app.db.models import InferenceLog, TaskStatus # THÊM MỚI
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
        logger.info(f"[Worker-Document] Start processing task {task_id}, page {page_order}")

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
        logger.info(f"[Worker-Document] Task {task_id}, page {page_order} processed. Remaining pages to process: {remaining_pages}")

        # Nếu không còn trang nào cần xử lý (counter <= 0), tiến hành tổng hợp kết quả
        if remaining_pages <= 0:
            logger.info(f"[Worker-Document] All pages for task {task_id} processed. Aggregating results.")
            await delete_counter(task_id)  # Dọn dẹp counter trong Redis

            combined_output = ''
            with get_session() as session:
                logs: list[InferenceLog] = get_logs_by_task(session=session, task_id=task_id)
                sorted_logs = sorted(logs, key=lambda log: log.page_order)

                for log in sorted_logs:
                    doc_block = extract_response(log.rawOutput).document
                    if log.objectUrls and doc_block:
                        image_map = dict(zip(list(log.objectKeys), list(log.objectUrls)))
                        doc_block, _ = replace_image_tags(doc_block, image_map)
                    
                    if doc_block:
                        combined_output += doc_block.strip() + "\n\n"
                
                update_task_result(session=session, result=combined_output.strip(), task_id=task_id)
                logger.info(f"[Worker-Document] Successfully wrote final result for task {task_id}")

    except Exception as e:
        logger.exception(f"[Worker-Document] FAILED to process task {task_id}, page {page_order}: {e}")
        # CẢI TIẾN: Cập nhật trạng thái task thành 'failed' để không bị treo
        with get_session() as session:
            task = get_task_by_id(session=session, task_id=task_id)
            if task and task.status == TaskStatus.pending:
                task.status = TaskStatus.failed
                task.error = f"Error on page {page_order}: {str(e)}"
                session.commit()
        # Xóa counter để các trang khác (nếu có) không bị kẹt
        await delete_counter(task_id)