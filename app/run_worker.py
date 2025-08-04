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
