"""CPU-only atomic-file protocol. No torch/CUDA imports in this module."""

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import socket
import tempfile
import time
import uuid


def atomic_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(value if isinstance(value, str) else json.dumps(value, sort_keys=True))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def read_json(path):
    return json.loads(Path(path).read_text())


@contextmanager
def exclusive(path):
    """A dead owner releases flock automatically; never unlink the lock inode."""
    with open(path, "a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def process_identity(pid=None):
    pid = os.getpid() if pid is None else int(pid)
    stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    if stat[0] == "Z":
        raise RuntimeError(f"Process {pid} is a zombie")
    return {"pid": pid, "start_ticks": stat[19], "host": socket.gethostname(),
            "boot_id": Path('/proc/sys/kernel/random/boot_id').read_text().strip()}


def validate_process(identity):
    if process_identity(identity["pid"]) != {k: identity[k] for k in
                                          ("pid", "start_ticks", "host", "boot_id")}:
        raise RuntimeError(f"Process identity changed: {identity}")


def steps(values):
    values = values.split() if isinstance(values, str) else values
    result = set()
    for value in values:
        if isinstance(value, bool) or not str(value).isdigit() or int(value) < 1:
            raise ValueError(f"Invalid suspend step: {value!r}")
        result.add(int(value))
    return sorted(result)


def submit(root, values):
    root = Path(root)
    read_json(root / "launch.json")  # Refuse a typo or uninitialized launch.
    with exclusive(root / "requests.lock"):
        path = root / "suspend_after_steps"
        old = steps(path.read_text()) if path.exists() else []
        merged = steps(old + steps(values))
        atomic_write(path, " ".join(map(str, merged)) + "\n")
    return merged


def claim(root):
    root = Path(root)
    path = root / "suspend_after_steps"
    with exclusive(root / "requests.lock"):
        if not path.exists():
            return None, []
        claimed = root / f"request.{uuid.uuid4().hex}.claimed"
        os.replace(path, claimed)
    try:
        return claimed, steps(claimed.read_text())
    except ValueError as error:
        atomic_write(claimed.with_suffix('.rejected.json'), {"error": str(error)})
        claimed.rename(claimed.with_suffix('.rejected'))
        return None, []


def matches(record, expected):
    return all(record.get(k) == expected[k] for k in ("job_id", "generation", "step"))


def wait_json(path, expected, interval, timeout=None):
    start = time.monotonic()
    while True:
        try:
            record = read_json(path)
            if matches(record, expected):
                return record
        except FileNotFoundError:
            pass
        if timeout is not None and time.monotonic() - start > timeout:
            raise TimeoutError(f"Waiting for {path}")
        time.sleep(interval)
