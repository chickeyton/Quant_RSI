"""Content-addressed artifacts; no pickle or executable model deserialization."""

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def decode(data: bytes) -> object:
    return json.loads(data, parse_constant=reject_constant)


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class ArtifactStore:
    def __init__(self, root: Path):
        self.root = root

    def put(self, data: bytes) -> str:
        key = digest(data)
        path = self.root / key
        if path.exists():
            self.get(key)
        else:
            atomic_write(path, data)
        return key

    def put_json(self, value: object) -> str:
        return self.put(canonical(value))

    def get(self, key: str) -> bytes:
        if not re.fullmatch(r"[0-9a-f]{64}", key):
            raise ValueError("invalid artifact reference")
        data = (self.root / key).read_bytes()
        if digest(data) != key:
            raise ValueError(f"artifact integrity failure: {key}")
        return data

    def json(self, key: str) -> object:
        return decode(self.get(key))
