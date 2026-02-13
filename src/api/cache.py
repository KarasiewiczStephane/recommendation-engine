"""Redis caching layer for recommendation results.

Provides async Redis cache operations with configurable TTL and
pattern-based invalidation for user and item-specific cache entries.
"""

import json
from typing import Any

import redis

from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RedisCache:
    """Redis-backed cache for recommendation and similarity results.

    Supports TTL-based expiration and pattern-based invalidation
    for efficient cache management.

    Attributes:
        client: The Redis client instance.
        rec_ttl: TTL for recommendation cache entries.
        sim_ttl: TTL for similarity cache entries.
    """

    def __init__(self, redis_client: redis.Redis) -> None:
        """Initialize the cache with a Redis client.

        Args:
            redis_client: A connected Redis client instance.
        """
        self.client = redis_client
        self.rec_ttl: int = config["redis"]["recommendation_ttl"]
        self.sim_ttl: int = config["redis"]["similarity_ttl"]

    def get(self, key: str) -> Any | None:
        """Retrieve a cached value by key.

        Args:
            key: The cache key.

        Returns:
            The deserialized cached value, or None if not found or on error.
        """
        try:
            data = self.client.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            logger.warning("Cache get error for key %s: %s", key, e)
        return None

    def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Store a value in the cache with TTL.

        Args:
            key: The cache key.
            value: The value to cache (must be JSON-serializable).
            ttl: Time-to-live in seconds. Defaults to recommendation TTL.

        Returns:
            True if the value was cached successfully.
        """
        try:
            self.client.set(key, json.dumps(value), ex=ttl or self.rec_ttl)
            return True
        except Exception as e:
            logger.warning("Cache set error for key %s: %s", key, e)
            return False

    def invalidate_user(self, user_id: int) -> int:
        """Invalidate all cached recommendations for a user.

        Args:
            user_id: The user whose cache entries should be removed.

        Returns:
            Number of cache entries deleted.
        """
        pattern = f"rec:{user_id}:*"
        keys = list(self.client.scan_iter(pattern))
        if keys:
            self.client.delete(*keys)
            logger.info("Invalidated %d cache entries for user %d", len(keys), user_id)
        return len(keys)

    def invalidate_item(self, item_id: int) -> int:
        """Invalidate cached similarity results for an item.

        Args:
            item_id: The item whose similarity cache should be cleared.

        Returns:
            Number of cache entries deleted.
        """
        pattern = f"sim:{item_id}:*"
        keys = list(self.client.scan_iter(pattern))
        if keys:
            self.client.delete(*keys)
        return len(keys)

    @staticmethod
    def rec_key(user_id: int, n: int, strategy: str) -> str:
        """Generate a cache key for recommendation results.

        Args:
            user_id: The user identifier.
            n: Number of recommendations.
            strategy: The recommendation strategy.

        Returns:
            Formatted cache key string.
        """
        return f"rec:{user_id}:{n}:{strategy}"

    @staticmethod
    def sim_key(item_id: int, n: int) -> str:
        """Generate a cache key for similarity results.

        Args:
            item_id: The item identifier.
            n: Number of similar items.

        Returns:
            Formatted cache key string.
        """
        return f"sim:{item_id}:{n}"
