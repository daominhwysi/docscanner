import asyncio
import json
from typing import Callable, Awaitable, Dict, Optional
from redis.asyncio import Redis
import os
redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = int(os.getenv("REDIS_PORT", 6379))
class SimpleRedisWorker:
    def __init__(self, queue_name: str = "tasks"):
        self.redis = Redis(host=redis_host, port=redis_port, decode_responses=True)
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
