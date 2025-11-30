import os
import config
import redis
import logging


REDIS_URL = config.REDIS_URL
CACHE_TTL_SECONDS = config.CACHE_TTL_SECONDS



# --- INITIALIZE REDIS CLIENT ---
try:
    redis_client = redis.Redis.from_url(REDIS_URL)
    redis_client.ping() # Test connection
    logging.info("Connected to Redis successfully.")
except Exception as e:
    logging.error(f"Failed to connect to Redis: {e}")
    redis_client = None
    
# key (query), value (answer from llm), ttl (time to live in seconds)
# Helper to format keys
def _key(k: str) -> str:
    return f"rag:cache:{k}"

# Get value from cache
def get(k: str):
    _value = redis_client.get(_key(k))
    return _value.decode() if _value else None

# Set value in cache with TTL
def set(k: str, v: str, ttl: int = CACHE_TTL_SECONDS):
    redis_client.setex(_key(k), ttl, v)
