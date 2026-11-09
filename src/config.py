"""Explicit, immutable experiment configuration. No command-line interface."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class DataConfig:
    url: str = "https://www.dropbox.com/s/vkzjokiwof5h8w6/afhq_v2.zip?dl=1"
    classes: tuple[str, ...] = ("cat", "dog", "wild")
    display_classes: tuple[str, ...] = ("cat", "dog", "wildlife")
    image_size: int = 128
    validation_fraction: float = 0.1
    split_seed: int = 1001
    expected_total: int = 15803


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 1
    seed: int = 2027
    microbatch: int = 16
    effective_batch: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    epsilon: float = 1e-8
    gradient_clip: float = 1.0
    warmup_epochs: int = 0
    minimum_lr_fraction: float = 0.1


@dataclass(frozen=True)
class TokenizerConfig:
    channels: tuple[int, ...] = (64, 128, 256)
    residual_blocks: int = 2
    codebook_size: int = 512
    embedding_dim: int = 64
    commitment_weight: float = 0.25
    perceptual_weight: float = 0.1
    training: TrainingConfig = field(default_factory=lambda: TrainingConfig(
        epochs=1, microbatch=4, learning_rate=2e-4, weight_decay=0.0, warmup_epochs=0,
    ))


@dataclass(frozen=True)
class DiffusionConfig:
    channels: tuple[int, ...] = (64, 128, 256)
    residual_blocks: int = 2
    attention_heads: int = 4
    attention_sizes: tuple[int, ...] = (8, 4)
    time_embedding_dim: int = 256
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    ema_decay: float = 0.999
    validation_seed: int = 4001
    training: TrainingConfig = field(default_factory=TrainingConfig)


@dataclass(frozen=True)
class AutoregressiveConfig:
    layers: int = 6
    width: int = 384
    heads: int = 6
    feedforward_dim: int = 1536
    dropout: float = 0.1
    training: TrainingConfig = field(default_factory=lambda: TrainingConfig(learning_rate=3e-4))


@dataclass(frozen=True)
class EvaluationConfig:
    max_per_class: int = 500
    sampling_seed: int = 5001
    reference_seed: int = 5002
    metric_seed: int = 5003
    ddim_steps: tuple[int, ...] = (25, 50, 100)
    temperatures: tuple[float, ...] = (0.8, 1.0, 1.2)
    generation_batch: int = 8
    feature_batch: int = 8
    kid_subsets: int = 100
    kid_class_subset: int = 100
    kid_overall_subset: int = 500
    precision_recall_k: int = 3
    benchmark_batches: tuple[int, ...] = (1, 8)
    benchmark_warmups: int = 3
    benchmark_repetitions: int = 30
    grid_per_class: int = 12


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "auto"
    num_workers: int = 0
    cpu_threads: int = 4
    io_workers: int = 8
    tokenizer_inference_batch: int = 32


@dataclass(frozen=True)
class ExperimentConfig:
    protocol_version: str = "1.0"
    data: DataConfig = field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    autoregressive: AutoregressiveConfig = field(default_factory=AutoregressiveConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @property
    def latent_size(self) -> int:
        return self.data.image_size // (2 ** len(self.tokenizer.channels))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.data.image_size != 128 or self.latent_size != 16:
            raise ValueError("This protocol requires 128px images and a 16x16 code grid.")
        if self.data.classes != ("cat", "dog", "wild"):
            raise ValueError("AFHQ archive classes must be cat/dog/wild.")
        if not 0 < self.data.validation_fraction < 1:
            raise ValueError("validation_fraction must be between zero and one.")
        if self.runtime.device not in ("auto", "mps", "cuda", "cpu"):
            raise ValueError("Unknown device.")
        if self.runtime.num_workers != 0 or self.runtime.cpu_threads < 1:
            raise ValueError("Use num_workers=0 and a positive CPU thread count.")
        if self.runtime.tokenizer_inference_batch < 1:
            raise ValueError("Tokenizer inference batch size must be positive.")
        if self.runtime.io_workers < 1:
            raise ValueError("Image preparation worker count must be positive.")
        if min(self.autoregressive.layers, self.autoregressive.width, self.autoregressive.heads,
               self.autoregressive.feedforward_dim, self.tokenizer.residual_blocks,
               self.tokenizer.codebook_size, self.tokenizer.embedding_dim,
               self.diffusion.residual_blocks, self.diffusion.attention_heads,
               self.diffusion.time_embedding_dim, self.diffusion.timesteps) < 1:
            raise ValueError("Model dimensions, depths, and heads must be positive.")
        if len(self.diffusion.channels) != 3 or any(c < 1 for c in (*self.tokenizer.channels, *self.diffusion.channels)):
            raise ValueError("Use three positive-channel diffusion levels and positive codec channels.")
        if self.autoregressive.width % self.autoregressive.heads:
            raise ValueError("Transformer width must divide evenly into attention heads.")
        if not 0 <= self.autoregressive.dropout < 1:
            raise ValueError("Invalid dropout.")
        for train in (self.tokenizer.training, self.diffusion.training, self.autoregressive.training):
            if min(train.epochs, train.microbatch, train.effective_batch) < 1:
                raise ValueError("Epoch and batch counts must be positive.")
            if train.effective_batch % train.microbatch:
                raise ValueError("effective_batch must be a multiple of microbatch.")
            if train.learning_rate <= 0 or train.gradient_clip <= 0 or train.weight_decay < 0:
                raise ValueError("Invalid optimizer settings.")
            if train.epsilon <= 0 or any(not 0 <= b < 1 for b in train.betas) or len(train.betas) != 2:
                raise ValueError("Invalid optimizer betas or epsilon.")
            if not 0 <= train.minimum_lr_fraction <= 1:
                raise ValueError("Invalid minimum learning-rate fraction.")
            if train.warmup_epochs < 0 or train.warmup_epochs >= train.epochs:
                raise ValueError("Warmup must be shorter than training.")
        if (self.diffusion.training.epochs, self.diffusion.training.effective_batch) != (
            self.autoregressive.training.epochs, self.autoregressive.training.effective_batch
        ):
            raise ValueError("The two priors must receive equal epochs and effective batches.")
        if not 0 < self.diffusion.beta_start < self.diffusion.beta_end < 1:
            raise ValueError("Invalid diffusion beta schedule.")
        if not 0 < self.diffusion.ema_decay < 1:
            raise ValueError("Invalid EMA decay.")
        if any(not 2 <= n <= self.diffusion.timesteps for n in self.evaluation.ddim_steps):
            raise ValueError("DDIM step count outside the training schedule.")
        if 50 not in self.evaluation.ddim_steps or 1.0 not in self.evaluation.temperatures:
            raise ValueError("Primary DDIM-50 and AR temperature 1.0 must be present.")
        if any(t <= 0 for t in self.evaluation.temperatures):
            raise ValueError("Temperatures must be positive.")
        ev = self.evaluation
        if min(ev.max_per_class, ev.generation_batch, ev.feature_batch, ev.kid_subsets,
               ev.precision_recall_k, ev.grid_per_class, *ev.benchmark_batches) < 1:
            raise ValueError("Evaluation sample, batch and neighborhood counts must be positive.")
        if min(ev.kid_class_subset, ev.kid_overall_subset, ev.benchmark_repetitions) < 2 or ev.benchmark_warmups < 0:
            raise ValueError("KID subset sizes and timing repetitions must be at least two; warmups nonnegative.")


def config_from_dict(value: dict[str, Any]) -> ExperimentConfig:
    """Round-trip JSON lists to immutable configuration tuples."""
    def construct(cls: type, raw: dict[str, Any], tuple_fields: tuple[str, ...] = ()) -> Any:
        raw = dict(raw)
        for key in tuple_fields:
            if key in raw:
                raw[key] = tuple(raw[key])
        if "training" in raw:
            raw["training"] = construct(TrainingConfig, raw["training"], ("betas",))
        return cls(**raw)

    result = ExperimentConfig(
        protocol_version=value["protocol_version"],
        data=construct(DataConfig, value["data"], ("classes", "display_classes")),
        tokenizer=construct(TokenizerConfig, value["tokenizer"], ("channels",)),
        diffusion=construct(DiffusionConfig, value["diffusion"], ("channels", "attention_sizes")),
        autoregressive=construct(AutoregressiveConfig, value["autoregressive"]),
        evaluation=construct(EvaluationConfig, value["evaluation"], ("ddim_steps", "temperatures", "benchmark_batches")),
        runtime=construct(RuntimeConfig, value["runtime"]),
    )
    result.validate()
    return result
