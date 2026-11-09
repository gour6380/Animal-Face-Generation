"""Analytical metric checks and protocol guards. Authored, not executed at delivery."""
import numpy as np
import pytest
import torch

from src.evaluate import (
    balanced_records,
    feature_precision_recall,
    kernel_inception_distance,
    nearest_training_examples,
)
from src.sample import image_to_uint8, setting_spec, verify_codec_metadata


def test_unbiased_kid_preserves_valid_negative_estimate():
    # For these two identical empirical sets, within-set excludes the diagonal;
    # the cross-set term includes it, giving the exact U-statistic -2.375.
    features = np.eye(2)
    result = kernel_inception_distance(features, features, subset_size=2, subsets=5, seed=9)
    assert result["mean"] == pytest.approx(-2.375)
    assert result["subset_std"] == pytest.approx(0.0)


def test_kid_rejects_insufficient_per_class_population():
    with pytest.raises(ValueError, match="subset"):
        kernel_inception_distance(np.zeros((20, 4)), np.zeros((20, 4)), subset_size=100, subsets=5, seed=0)


def test_kid_subset_randomness_is_local_and_reproducible():
    real = np.arange(120, dtype=float).reshape(30, 4) / 120
    fake = real[::-1] + 0.1
    one = kernel_inception_distance(real, fake, subset_size=10, subsets=12, seed=71)
    np.random.seed(991)
    two = kernel_inception_distance(real, fake, subset_size=10, subsets=12, seed=71)
    assert one == two


def test_balanced_holdout_is_reproducible_disjoint_and_without_replacement():
    records = [{"id": f"{label}_{index}", "label": label} for label in range(3) for index in range(10 + label)]
    first = balanced_records(records, n_per_class=5, seed=42)
    assert first == balanced_records(list(reversed(records)), n_per_class=5, seed=42)
    assert len({row["id"] for row in first}) == 15
    assert [sum(row["label"] == label for row in first) for label in range(3)] == [5, 5, 5]
    with pytest.raises(ValueError):
        balanced_records(records, n_per_class=11, seed=42)


def test_precision_recall_distinguishes_matching_and_separated_manifolds():
    features = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    matching = feature_precision_recall(features, features, k=1)
    separated = feature_precision_recall(features, features + 100, k=1)
    assert matching["precision"] == 1 and matching["recall"] == 1
    assert separated["precision"] == 0 and separated["recall"] == 0


def test_nearest_training_search_has_expected_matches_across_chunks():
    train = torch.tensor([[0., 0.], [10., 10.], [20., 20.]])
    samples = torch.tensor([[19., 20.], [0., 1.], [11., 10.]])
    indexes, distances = nearest_training_examples(samples, train, chunk=1)
    assert indexes == [2, 0, 1]
    assert distances == pytest.approx([1., 1., 1.])


def test_uint8_conversion_clamps_and_rounds_consistently():
    converted = image_to_uint8(torch.tensor([-2., -1., 0., 1., 2.]))
    assert converted.dtype == torch.uint8 and converted.device.type == "cpu"
    assert converted.tolist() == [0, 0, 128, 255, 255]


@pytest.mark.parametrize("family,setting", [("diffusion", "temperature_1.0"),
                                           ("autoregressive", "ddim_50"),
                                           ("autoregressive", "temperature_nan"),
                                           ("diffusion", "ddim_1")])
def test_setting_names_cannot_silently_change_sampler(family, setting):
    with pytest.raises(ValueError):
        setting_spec(family, setting)


@pytest.mark.parametrize("key,value", [("codec_sha256", "other"), ("mean", 0.2),
                                      ("std", 0.7), ("token_cache_identity", "other-cache")])
def test_prior_loader_rejects_codec_or_normalization_mismatch(key, value):
    codec = {"codec_sha256": "frozen", "mean": 0.0, "std": 1.0, "token_cache_identity": "cache"}
    changed = {**codec, key: value}
    with pytest.raises(RuntimeError, match=key):
        verify_codec_metadata(changed, codec)
