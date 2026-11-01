"""Process-wide TTL cache for fetched market data and analysis decisions."""
import threading
import time

_lock = threading.Lock()
_entries = {}  # (kind, key) -> (expires_at, value, version)
_versions = {}  # (kind, key) -> last version handed out


def cached(kind, key, ttl):
    """Return (value, version) for a fresh entry, or None when absent or expired."""
    if ttl <= 0:
        return None
    with _lock:
        entry = _entries.get((kind, key))
        if entry is not None and entry[0] > time.monotonic():
            return entry[1], entry[2]
    return None


def store(kind, key, value, ttl):
    """Store a value and return its version, or None when ttl is not positive."""
    if ttl <= 0:
        return None
    with _lock:
        now = time.monotonic()
        for slot in [slot for slot, entry in _entries.items() if entry[0] <= now]:
            del _entries[slot]
        slot = (kind, key)
        version = _versions.get(slot, 0) + 1
        _versions[slot] = version
        _entries[slot] = (now + ttl, value, version)
        return version


def clear():
    # Versions keep increasing so a slot's version never repeats, even after a clear.
    with _lock:
        _entries.clear()
