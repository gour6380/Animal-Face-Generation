"""Process-level run gate: concurrent family readers, exclusive coordinators."""
from __future__ import annotations

import fcntl
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO

from filelock import Timeout


class RunGate:
    """Reentrant for one Run context; independent contexts still compete."""

    def __init__(self, path: Path):
        self.path = path
        self._stream: TextIO | None = None
        self._shared = False
        self._depth = 0

    @contextmanager
    def acquire(self, *, shared: bool = False) -> Iterator[None]:
        if self._depth:
            if self._shared and not shared:
                raise Timeout(str(self.path))
            self._depth += 1
        else:
            stream = self.path.open("a+")
            try:
                fcntl.flock(stream, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
            except BlockingIOError as error:
                stream.close()
                raise Timeout(str(self.path)) from error
            self._stream, self._shared, self._depth = stream, shared, 1
        try:
            yield
        finally:
            self._depth -= 1
            if not self._depth:
                assert self._stream is not None
                fcntl.flock(self._stream, fcntl.LOCK_UN)
                self._stream.close()
                self._stream = None
