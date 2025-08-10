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