"""Content-addressed intraday bar storage with deterministic bytes.

Each block holds one instrument, grid and session. Columns are canonical little-endian bytes
with no container format (`np.savez` embeds zip timestamps and would break digests). A
canonical JSON manifest binds column digests, the data contract, raw responses and receipts.
"""

import os
import re
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from quant_rsi.core.artifacts import canonical, decode, digest
from quant_rsi.intraday.contracts import IntradayDataContract

VALID, MISSING, INVALID, STALE = 0, 1, 2, 3
GRIDS = ("decision", "execution", "execution_quotes")
BAR_COLUMNS = {
    "start": "<i8",
    "open": "<f8",
    "high": "<f8",
    "low": "<f8",
    "close": "<f8",
    "volume": "<f8",
    "status": "|u1",
}
QUOTE_COLUMNS = {"start": "<i8", "bid": "<f8", "ask": "<f8", "status": "|u1"}


def columns_for(grid: str) -> dict[str, str]:
    if grid not in GRIDS:
        raise ValueError(f"unknown grid {grid}")
    return QUOTE_COLUMNS if grid == "execution_quotes" else BAR_COLUMNS


def portable_atomic_write(path: Path, data: bytes) -> None:
    """Atomic replace; directory fsync where the platform supports it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        flag = getattr(os, "O_DIRECTORY", None)
        if flag is not None:
            directory = os.open(path.parent, os.O_RDONLY | flag)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def encode_column(values: np.ndarray, dtype: str) -> bytes:
    array = np.ascontiguousarray(np.asarray(values).astype(np.dtype(dtype), copy=False))
    if array.ndim != 1:
        raise ValueError("columns are one-dimensional")
    return array.tobytes()


def decode_column(data: bytes, dtype: str, length: int) -> np.ndarray:
    array = np.frombuffer(data, dtype=np.dtype(dtype))
    if array.shape != (length,):
        raise ValueError("column length does not match its manifest")
    return array.copy()


@dataclass(frozen=True)
class Block:
    instrument: str
    grid: str
    session_id: str
    columns: Mapping[str, np.ndarray]

    def __post_init__(self):
        expected = columns_for(self.grid)
        if set(self.columns) != set(expected):
            raise ValueError(f"{self.grid} block needs columns {sorted(expected)}")
        lengths = {len(v) for v in self.columns.values()}
        if len(lengths) != 1:
            raise ValueError("block columns must share one length")
        status = np.asarray(self.columns["status"])
        if not np.isin(status, (VALID, MISSING, INVALID, STALE)).all():
            raise ValueError("unknown bar status")
        prices = [c for c in self.columns if c not in {"start", "status", "volume"}]
        for name in prices:
            values = np.asarray(self.columns[name], dtype=float)
            valid = status == VALID
            if not (np.isfinite(values[valid]).all() and (values[valid] > 0).all()):
                raise ValueError("valid bars need finite positive prices")
            if np.isfinite(values[~valid]).any():
                raise ValueError("missing bars must not carry prices (no fill-forward)")

    def __len__(self) -> int:
        return len(self.columns["start"])


def aggregate_execution(execution: Block, starts: np.ndarray, bar_seconds: int) -> Block:
    """Decision bars built from 1-minute execution bars on the registered decision grid.

    A decision bar is valid when at least one of its minutes is valid: open of the first
    valid minute, close of the last, max high, min low and summed volume. A bar with no valid
    minute is missing; nothing is forward-filled.
    """
    minute = execution.columns
    count = len(starts)
    out = {k: np.full(count, np.nan) for k in ("open", "high", "low", "close", "volume")}
    status = np.full(count, MISSING, np.uint8)
    for i, start in enumerate(starts):
        inside = (minute["start"] >= start) & (minute["start"] < start + bar_seconds)
        valid = np.flatnonzero(inside & (minute["status"] == VALID))
        if not len(valid):
            if (minute["status"][inside] == INVALID).any():
                status[i] = INVALID
            continue
        out["open"][i], out["close"][i] = minute["open"][valid[0]], minute["close"][valid[-1]]
        out["high"][i], out["low"][i] = minute["high"][valid].max(), minute["low"][valid].min()
        out["volume"][i] = minute["volume"][valid].sum()
        status[i] = VALID
    return Block(
        execution.instrument,
        "decision",
        execution.session_id,
        {"start": np.asarray(starts, dtype=np.int64), **out, "status": status},
    )


class SnapshotStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{64}", key):
            raise ValueError("invalid artifact reference")
        return self.root / "objects" / key[:2] / key

    def put(self, data: bytes) -> str:
        key = digest(data)
        path = self._path(key)
        if path.exists():
            self.get(key)
        else:
            portable_atomic_write(path, data)
        return key

    def get(self, key: str) -> bytes:
        data = self._path(key).read_bytes()
        if digest(data) != key:
            raise ValueError(f"artifact integrity failure: {key}")
        return data

    def put_raw(self, name: str, body: bytes, request: Mapping[str, object]) -> str:
        """Raw provider bytes plus the request that produced them, before normalisation."""
        envelope = {"name": name, "request": request, "body_ref": self.put(body)}
        return self.put(canonical(envelope))

    def _request_path(self, request: Mapping[str, object]) -> Path:
        return self.root / "requests" / digest(canonical(dict(request)))

    def find_raw(self, request: Mapping[str, object]) -> tuple[list, dict] | None:
        """Previously stored payloads and original receipt for an identical request."""
        path = self._request_path(request)
        if not path.exists():
            return None
        receipt = decode(path.read_bytes())
        envelope = decode(self.get(receipt["raw_ref"]))
        if envelope["request"] != decode(canonical(dict(request))):
            raise ValueError("request index entry does not match its stored request")
        return decode(self.get(envelope["body_ref"])), receipt

    def remember_raw(self, request: Mapping[str, object], receipt: Mapping[str, object]) -> None:
        portable_atomic_write(self._request_path(request), canonical(dict(receipt)))

    def put_block(self, block: Block) -> dict:
        expected = columns_for(block.grid)
        columns = {}
        for name, dtype in sorted(expected.items()):
            data = encode_column(block.columns[name], dtype)
            columns[name] = {"dtype": dtype, "length": len(block), "sha256": self.put(data)}
        return {
            "instrument": block.instrument,
            "grid": block.grid,
            "session_id": block.session_id,
            "columns": columns,
        }

    def commit(
        self,
        contract: IntradayDataContract,
        blocks: Iterable[Block],
        *,
        raw_refs: Iterable[str] = (),
        receipts: Iterable[Mapping[str, object]] = (),
    ) -> str:
        entries = sorted(
            (self.put_block(block) for block in blocks),
            key=lambda e: (e["instrument"], e["grid"], e["session_id"]),
        )
        keys = [(e["instrument"], e["grid"], e["session_id"]) for e in entries]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate block in snapshot")
        manifest = {
            "schema_version": 1,
            "kind": "IntradaySnapshot",
            "contract": contract.model_dump(mode="json"),
            "blocks": entries,
            "raw_refs": sorted(set(raw_refs)),
            "receipts": sorted((dict(r) for r in receipts), key=canonical),
        }
        return self.put(canonical(manifest))

    def load_block(self, entry: Mapping[str, object]) -> Block:
        """Rebuild a block from a `put_block` entry, verifying every column digest."""
        columns = {
            name: decode_column(self.get(spec["sha256"]), spec["dtype"], spec["length"])
            for name, spec in entry["columns"].items()
        }
        return Block(entry["instrument"], entry["grid"], entry["session_id"], columns)

    def load(self, snapshot_ref: str) -> "Snapshot":
        manifest = decode(self.get(snapshot_ref))
        if manifest.get("kind") != "IntradaySnapshot":
            raise ValueError("not an intraday snapshot")
        contract = IntradayDataContract.model_validate_json(canonical(manifest["contract"]))
        blocks = {}
        for entry in manifest["blocks"]:
            block = self.load_block(entry)
            blocks[(block.instrument, block.grid, block.session_id)] = block
        return Snapshot(snapshot_ref, contract, blocks, tuple(manifest["receipts"]))


@dataclass(frozen=True)
class Snapshot:
    ref: str
    contract: IntradayDataContract
    blocks: Mapping[tuple[str, str, str], Block]
    receipts: tuple = field(default=())

    def block(self, instrument: str, grid: str, session_id: str) -> Block | None:
        return self.blocks.get((instrument, grid, session_id))
