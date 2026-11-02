import hashlib
import hmac
import secrets
import threading
import time
from collections import OrderedDict, deque

from fastapi import HTTPException


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=16384, r=8, p=1, dklen=32)
    return f"scrypt${salt.hex()}${digest.hex()}"


def check_password(password: str, stored: str) -> bool:
    try:
        algorithm, salt, expected = stored.split("$")
        if algorithm != "scrypt":
            return False
        actual = hashlib.scrypt(
            password.encode(), salt=bytes.fromhex(salt), n=16384, r=8, p=1, dklen=32
        )
        return hmac.compare_digest(actual, bytes.fromhex(expected))
    except (ValueError, TypeError):
        return False


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class RateLimiter:
    """Bounded process-local limiter. Put a shared limiter at ingress for replicas."""

    def __init__(self, capacity=4096):
        self.capacity = capacity
        self.lock = threading.Lock()
        self.buckets = OrderedDict()

    def check(self, key, limit, seconds):
        now = time.monotonic()
        with self.lock:
            bucket = self.buckets.get(key)
            if bucket is None:
                if len(self.buckets) >= self.capacity:
                    # Fail closed while the oldest key still has a live window.
                    oldest = next(iter(self.buckets))
                    old_times, old_window = self.buckets[oldest]
                    if old_times and now - old_times[-1] < old_window:
                        raise HTTPException(429, "Rate limiter capacity reached")
                    self.buckets.pop(oldest)
                bucket = (deque(), seconds)
                self.buckets[key] = bucket
            times, _ = bucket
            self.buckets.move_to_end(key)
            while times and times[0] <= now - seconds:
                times.popleft()
            if len(times) >= limit:
                raise HTTPException(
                    429, "Too many requests", headers={"Retry-After": str(seconds)}
                )
            times.append(now)
