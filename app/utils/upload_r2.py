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