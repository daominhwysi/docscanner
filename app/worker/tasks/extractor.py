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
from app.db.models import InferenceLog, TaskStatus, Task
from app.services.logging_config import get_logger
import re
logger = get_logger()
AML_CLOSE_TAG = re.compile(r"</AssessmentMarkupLanguage\s*>", re.IGNORECASE)


async def uploadImageFromUrls(ImageUrls) -> List[str]:
    if not ImageUrls:
        return []
    image_bytes = await asyncio.gather(*(get_file_bytes(img) for img in ImageUrls))
    image_urls = await upload_multiple_images(image_bytes, concurrency_limit=10)
    return image_urls

async def call_gemini_with_retry(contents, config, max_retries=2) -> types.GenerateContentResponse:
    llmResponse = None
    last_response_text = ""

    for attempt in range(max_retries):
        llmResponse = await GeminiAgent(model="gemini-2.0-flash-001", contents=contents, config=config)
        if llmResponse and llmResponse.text:
            last_response_text = llmResponse.text
            if AML_CLOSE_TAG.search(last_response_text):
                return llmResponse
        logger.warning(f"Attempt {attempt+1}/{max_retries} missing </AssessmentMarkupLanguage>, retrying...")

    # Fallback: thêm thẻ đóng vào cuối nếu chưa có
    if last_response_text and not AML_CLOSE_TAG.search(last_response_text):
        logger.info("Fallback: appending </AssessmentMarkupLanguage> at the end")
        last_response_text = last_response_text.rstrip() + "\n</AssessmentMarkupLanguage>"
        if llmResponse:
            llmResponse.text = last_response_text
        else:
            llmResponse = types.GenerateContentResponse(text=last_response_text)

    return llmResponse


async def convert_to_webp_base64(img_bytes: bytes, quality: int = 80) -> str:
    with Image.open(BytesIO(img_bytes)) as img:
        output = BytesIO()
        img.convert("RGBA").save(output, format="WEBP", quality=quality, method=6)
        webp_bytes = output.getvalue()
        base64_str = base64.b64encode(webp_bytes).decode('utf-8')
        return base64_str


def create_error_log(task_id: str, page_order: int, error_message: str, img_url: str = None):
    """Tạo log entry cho trang bị lỗi"""
    try:
        with get_session() as session:
            create_log(
                imageUrls=img_url or "",
                objectKeys=[],
                objectUrls=[],
                requestId=task_id,
                num_input_token=0,
                num_output_token=0,
                rawOutput=None,  # None để đánh dấu trang lỗi
                page_order=page_order,
                error=error_message,  # Thêm trường error vào create_log function
                session=session
            )
            logger.info(f"[{task_id}] Created error log for page {page_order}")
    except Exception as log_error:
        logger.error(f"[{task_id}] Failed to create error log for page {page_order}: {log_error}")


async def aggregate_results(task_id: str):
    """Tổng hợp kết quả từ tất cả các trang đã xử lý"""
    try:
        with get_session() as session:
            # Lock task để tránh race condition
            task = session.query(Task).filter(Task.id == task_id).with_for_update().first()
            if not task:
                logger.error(f"[{task_id}] Task not found during aggregation")
                return
                
            if task.status != TaskStatus.pending:
                logger.warning(f"[{task_id}] Task is no longer in 'pending' state (current: {task.status.value}). Skipping aggregation.")
                return

            # Lấy tất cả logs của task
            logs: list[InferenceLog] = get_logs_by_task(session=session, task_id=task_id)
            sorted_logs = sorted(logs, key=lambda log: log.page_order)
            
            # Phân loại logs thành thành công và lỗi
            successful_logs = []
            error_logs = []
            
            for log in sorted_logs:
                if log.rawOutput is None:  # Trang bị lỗi
                    error_logs.append(log)
                else:
                    successful_logs.append(log)
            
            # Tổng hợp nội dung từ các trang thành công
            combined_output = ''
            for log in successful_logs:
                doc_block = extract_response(log.rawOutput).document
                if log.objectUrls and doc_block:
                    image_map = dict(zip(list(log.objectKeys), list(log.objectUrls)))
                    doc_block, _ = replace_image_tags(doc_block, image_map)
                
                if doc_block:
                    combined_output += doc_block.strip() + "\n"
            
            # Quyết định trạng thái cuối cùng của task
            if error_logs:
                error_summary = f"Errors occurred on {len(error_logs)} page(s): " + \
                              ", ".join([f"page {log.page_order}" for log in error_logs])
                
                if successful_logs:
                    # Có cả trang thành công và trang lỗi
                    final_result = combined_output.strip()
                    if final_result:
                        final_result += f"\n\n[WARNING: {error_summary}]"
                    else:
                        final_result = f"[ERROR: {error_summary}]"
                    
                    # Cập nhật task với trạng thái done nhưng có cảnh báo
                    update_task_result(session=session, result=final_result, task_id=task_id)
                    task.status = TaskStatus.done  # Hoặc tạo TaskStatus.partial_success nếu cần
                    logger.warning(f"[{task_id}] Task completed with warnings: {error_summary}")
                else:
                    # Tất cả trang đều lỗi
                    task.status = TaskStatus.failed
                    task.error = error_summary
                    logger.error(f"[{task_id}] Task failed: all pages had errors")
            else:
                # Tất cả trang đều thành công
                update_task_result(session=session, result=combined_output.strip(), task_id=task_id)
                logger.info(f"[{task_id}] Task completed successfully")
            
            session.commit()
            
    except Exception as e:
        logger.exception(f"[{task_id}] Failed to aggregate results: {e}")
        # Trong trường hợp lỗi nghiêm trọng, đánh dấu task failed
        try:
            with get_session() as session:
                task = get_task_by_id(session=session, task_id=task_id)
                if task and task.status == TaskStatus.pending:
                    task.status = TaskStatus.failed
                    task.error = f"Aggregation failed: {str(e)}"
                    session.commit()
        except Exception as final_error:
            logger.error(f"[{task_id}] Failed to update task status after aggregation error: {final_error}")


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
                call_gemini_with_retry(contents=contents, config=generate_content_config),
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
            llmResponse = await call_gemini_with_retry(contents=contents, config=generate_content_config)

        if not llmResponse or not llmResponse.text:
            raise ValueError("LLM response is invalid or empty!")

        # Tạo log entry cho trang thành công
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
                error=None,  # Không có lỗi
                session=session
            )
        
        logger.info(f"[{task_id}] Successfully processed page {page_order}")

    except Exception as e:
        # Xử lý lỗi: tạo error log thay vì fail toàn bộ task
        logger.exception(f"[{task_id}] Error processing page {page_order}: {e}")
        create_error_log(task_id=task_id, page_order=page_order, error_message=str(e), img_url=img_url)

    finally:
        # Logic counter và tổng hợp kết quả (chạy dù có lỗi hay không)
        try:
            remaining_pages = await decrement_counter(task_id)
            logger.info(f"[{task_id}] Page {page_order} finished. Remaining pages: {remaining_pages}")

            # Nếu đây là trang cuối cùng, tiến hành tổng hợp
            if remaining_pages <= 0:
                logger.info(f"[{task_id}] All pages processed. Starting result aggregation.")
                await delete_counter(task_id)
                await aggregate_results(task_id)
                
        except Exception as counter_error:
            logger.error(f"[{task_id}] Error in counter logic for page {page_order}: {counter_error}")
            try:
                await delete_counter(task_id)
            except:
                pass