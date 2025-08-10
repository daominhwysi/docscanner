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
