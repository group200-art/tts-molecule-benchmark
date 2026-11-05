"""Phase 1A helpers: stdlib only; new, no-clobber outputs inside BenchmarkV2."""
import hashlib
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def new_file(relative_path, data):
    """Atomic publication without replacing any existing target, including symlinks."""
    target = ROOT / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if ROOT not in target.resolve().parents:
        raise ValueError("Output must be strictly inside BenchmarkV2")
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    temporary_dir = ROOT / "tmp"
    temporary_dir.mkdir(exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="phase1a_", dir=str(temporary_dir))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data.encode() if isinstance(data, str) else data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(name, target)  # fails atomically if another session created target
    finally:
        os.unlink(name)
    return target


def new_json(path, value):
    return new_file(path, json_bytes(value))


def jsonl_bytes(rows):
    return "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows)


def iter_json_array(handle, chunk_size=1048576):
    """Stream a JSON array; reject truncation, trailing commas and trailing data."""
    decoder = json.JSONDecoder()
    buffer, pos, eof = "", 0, False

    def fill():
        nonlocal buffer, pos, eof
        chunk = handle.read(chunk_size)
        buffer = buffer[pos:] + chunk
        pos = 0
        eof = not chunk

    def whitespace():
        nonlocal pos
        while True:
            while pos < len(buffer) and buffer[pos].isspace():
                pos += 1
            if pos < len(buffer) or eof:
                return
            fill()

    whitespace()
    if pos >= len(buffer) or buffer[pos] != "[":
        raise ValueError("Expected JSON array")
    pos += 1
    after_comma = False
    while True:
        whitespace()
        if pos >= len(buffer):
            raise ValueError("Truncated JSON array")
        if buffer[pos] == "]":
            if after_comma:
                raise ValueError("Trailing comma")
            pos += 1
            break
        while True:
            try:
                record, end = decoder.raw_decode(buffer, pos)
                # A numeric scalar may end at a chunk boundary. Fetch its delimiter.
                if end == len(buffer) and not eof:
                    fill()
                    continue
                pos = end
                break
            except json.JSONDecodeError:
                if eof:
                    raise
                fill()
        yield record
        whitespace()
        if pos >= len(buffer):
            raise ValueError("Missing array terminator")
        if buffer[pos] == ",":
            pos += 1
            after_comma = True
        elif buffer[pos] == "]":
            after_comma = False
        else:
            raise ValueError("Expected comma or closing bracket")
    whitespace()
    if pos < len(buffer):
        raise ValueError("Trailing JSON data")
