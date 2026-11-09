"""TensorBoard visualization and text logs; checkpoints own canonical metrics."""
from __future__ import annotations

import logging
from numbers import Real
from types import TracebackType
from typing import Any, Self

from torch.utils.tensorboard import SummaryWriter

from .runs import Run


class TrainingTracker:
    def __init__(self, run: Run, family: str, purge_step: int | None = None):
        self.writer = SummaryWriter(log_dir=str(run.path / "tensorboard" / family), purge_step=purge_step)
        self.logger = logging.getLogger(f"animal_generation.{run.path.name}.{family}.{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.handler = logging.FileHandler(run.path / "logs" / f"{family}.log")
        self.handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        self.logger.addHandler(self.handler)

    def log(self, epoch: int, metrics: dict[str, Any]) -> None:
        for name, value in metrics.items():
            if isinstance(value, Real):
                self.writer.add_scalar(name, float(value), epoch)
        self.writer.flush()
        self.logger.info("epoch=%s metrics=%s", epoch, metrics)

    def close(self) -> None:
        self.writer.close()
        self.handler.close()
        self.logger.removeHandler(self.handler)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None,
                 traceback: TracebackType | None) -> None:
        if exc_value is not None:
            self.logger.error("Training interrupted: %s", exc_value)
        self.close()


def launch_tensorboard(run: Run) -> str:
    """Explicit optional notebook operation; bind the dashboard to localhost."""
    from tensorboard import program
    server = program.TensorBoard()
    server.configure(argv=["tensorboard", "--logdir", str(run.path / "tensorboard"), "--host", "127.0.0.1", "--port", "0"])
    return server.launch()
