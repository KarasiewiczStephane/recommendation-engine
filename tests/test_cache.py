"""Tests for the Redis caching layer."""

import fakeredis
import pytest

from src.api.cache import RedisCache


@pytest.fixture
def redis_client() -> fakeredis.FakeRedis:
    """Create a fake Redis client for testing."""
    client = fakeredis.FakeRedis(decode_responses=True)
    yield client
    client.flushall()


@pytest.fixture
def cache(redis_client: fakeredis.FakeRedis) -> RedisCache:
    """Create a RedisCache with a fake Redis backend."""
    return RedisCache(redis_client)


class TestRedisCache:
    """Tests for the RedisCache class."""

    def test_set_and_get(self, cache: RedisCache) -> None:
        """Stored values can be retrieved."""
        cache.set("test_key", {"value": 42})
        result = cache.get("test_key")
        assert result == {"value": 42}

    def test_get_missing_key(self, cache: RedisCache) -> None:
        """Missing keys return None."""
        result = cache.get("nonexistent")
        assert result is None

    def test_set_with_ttl(
        self, cache: RedisCache, redis_client: fakeredis.FakeRedis
    ) -> None:
        """Values are set with the specified TTL."""
        cache.set("ttl_key", "data", ttl=60)
        ttl = redis_client.ttl("ttl_key")
        assert ttl > 0
        assert ttl <= 60

    def test_set_returns_true(self, cache: RedisCache) -> None:
        """Successful set returns True."""
        assert cache.set("key", "value") is True

    def test_invalidate_user(self, cache: RedisCache) -> None:
        """User cache invalidation removes matching keys."""
        cache.set("rec:1:10:weighted", {"data": 1})
        cache.set("rec:1:5:switching", {"data": 2})
        cache.set("rec:2:10:weighted", {"data": 3})

        deleted = cache.invalidate_user(1)
        assert deleted == 2
        assert cache.get("rec:1:10:weighted") is None
        assert cache.get("rec:2:10:weighted") is not None

    def test_invalidate_user_no_keys(self, cache: RedisCache) -> None:
        """User invalidation with no matching keys returns 0."""
        deleted = cache.invalidate_user(999)
        assert deleted == 0

    def test_invalidate_item(self, cache: RedisCache) -> None:
        """Item cache invalidation removes matching keys."""
        cache.set("sim:1:10", {"data": 1})
        cache.set("sim:1:5", {"data": 2})
        cache.set("sim:2:10", {"data": 3})

        deleted = cache.invalidate_item(1)
        assert deleted == 2
        assert cache.get("sim:1:10") is None

    def test_rec_key_format(self) -> None:
        """Recommendation key follows expected format."""
        key = RedisCache.rec_key(user_id=1, n=10, strategy="weighted")
        assert key == "rec:1:10:weighted"

    def test_sim_key_format(self) -> None:
        """Similarity key follows expected format."""
        key = RedisCache.sim_key(item_id=42, n=5)
        assert key == "sim:42:5"

    def test_cache_list_data(self, cache: RedisCache) -> None:
        """Complex data structures round-trip correctly."""
        data = [{"item_id": 1, "score": 0.95}, {"item_id": 2, "score": 0.87}]
        cache.set("complex", data)
        result = cache.get("complex")
        assert result == data

    def test_default_ttl_applied(
        self, cache: RedisCache, redis_client: fakeredis.FakeRedis
    ) -> None:
        """Default recommendation TTL is applied when no TTL specified."""
        cache.set("default_ttl", "value")
        ttl = redis_client.ttl("default_ttl")
        assert ttl > 0
