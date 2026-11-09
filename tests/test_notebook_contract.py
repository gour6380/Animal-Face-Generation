"""Inspect the notebook interface without executing any notebook cell."""
from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "animal_face_generation_public.ipynb"


def test_output_free_portable_notebook_and_explicit_worker_actions():
    notebook = json.loads(NOTEBOOK.read_text())
    assert "widgets" not in notebook.get("metadata", {})
    execution = ast.parse((ROOT / "src" / "execution.py").read_text())
    actions = next(ast.literal_eval(node.value) for node in execution.body
                   if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                   and node.targets[0].id == "_ACTIONS")
    text = ""
    operation_order = []
    for cell in notebook["cells"]:
        assert not cell.get("attachments")
        if cell["cell_type"] != "code":
            continue
        assert cell["outputs"] == []
        assert cell["execution_count"] is None
        source = "".join(cell["source"])
        text += source
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if not isinstance(node.func.value, ast.Name) or node.func.value.id != "launcher":
                continue
            operation_order.append(node.func.attr)
            if node.func.attr == "run_action":
                assert node.args[0].value in actions
    assert 'sys.path.insert' not in text
    assert 'sys.prefix' not in text
    assert '/Users/' not in text
    assert '"-m", "pip", "install"' in text
    assert text.index('"-m", "pip", "install"') < text.index('from src.config')
    assert operation_order.count('start_training') == 2
    assert operation_order.count('wait') == 1
    assert text.index('launcher.wait') < text.index('"freeze_evaluation"')
    assert notebook['metadata']['kernelspec']['name'] == 'python3'


def test_all_registered_notebook_functions_exist_with_expected_keywords():
    """Static signature check catches misspelled worker keyword arguments."""
    module = ast.parse((ROOT / "src" / "execution.py").read_text())
    actions = next(ast.literal_eval(node.value) for node in module.body
                   if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                   and node.targets[0].id == "_ACTIONS")
    notebook = json.loads(NOTEBOOK.read_text())
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        for node in ast.walk(ast.parse(''.join(cell['source']))):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'run_action'):
                continue
            action = node.args[0].value
            implementation = ast.parse((ROOT / 'src' / f'{actions[action]}.py').read_text())
            function = next(item for item in implementation.body if isinstance(item, ast.FunctionDef) and item.name == action)
            names = {arg.arg for arg in [*function.args.args, *function.args.kwonlyargs]}
            assert {keyword.arg for keyword in node.keywords} <= names


def _notebook_configuration_literals():
    """Read only literal configuration assignments; never execute notebook code."""
    constructors = {
        "data_config": "DataConfig", "tokenizer_config": "TokenizerConfig",
        "diffusion_config": "DiffusionConfig", "autoregressive_config": "AutoregressiveConfig",
        "evaluation_config": "EvaluationConfig", "runtime_config": "RuntimeConfig",
        "experiment_config": "ExperimentConfig",
    }
    values = {}

    def extract(node):
        if isinstance(node, ast.Name):
            return values[node.id]
        if isinstance(node, ast.Call):
            assert isinstance(node.func, ast.Name)
            assert node.func.id in {*constructors.values(), "TrainingConfig"}
            assert not node.args
            assert all(keyword.arg is not None for keyword in node.keywords)
            return {keyword.arg: extract(keyword.value) for keyword in node.keywords}
        return ast.literal_eval(node)

    notebook = json.loads(NOTEBOOK.read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)):
                continue
            name = node.targets[0].id
            if name not in constructors:
                continue
            assert name not in values, "Each configuration has one visible definition."
            assert len(ast.parse("".join(cell["source"])).body) == 1
            assert isinstance(node.value, ast.Call)
            assert isinstance(node.value.func, ast.Name)
            assert node.value.func.id == constructors[name]
            values[name] = extract(node.value)
    assert values.keys() == constructors.keys()
    return values


def test_l40s_preset_is_valid_with_equal_exposure_and_full_evaluation():
    from src.config import config_from_dict

    configs = _notebook_configuration_literals()
    assert configs["experiment_config"] == {
        name: configs[f"{name}_config"] for name in (
            "data", "tokenizer", "diffusion", "autoregressive", "evaluation", "runtime"
        )
    }
    config = config_from_dict({"protocol_version": "1.0", **configs["experiment_config"]})
    config.validate()
    # The public reproduction notebook must match the published measured recipe.
    recorded = config_from_dict(json.loads((ROOT / "reports" / "run_001" / "run_config.json").read_text()))
    assert config == recorded
    trainings = (config.tokenizer.training, config.diffusion.training, config.autoregressive.training)
    assert tuple((training.epochs, training.effective_batch, training.warmup_epochs)
                 for training in trainings) == ((100, 64, 0), (200, 64, 0), (200, 64, 0))
    # A rank's microbatch must fit an integral number of times into an update
    # with either one or two participating tokenizer ranks.
    assert all(training.effective_batch % (ranks * training.microbatch) == 0
               for training in trainings for ranks in (1, 2))
    assert tuple(training.microbatch for training in trainings) == (8, 32, 32)
    ev = config.evaluation
    assert (config.runtime.tokenizer_inference_batch, ev.generation_batch, ev.feature_batch) == (64, 16, 32)
    assert ev.ddim_steps == (25, 50, 100)
    assert ev.temperatures == (0.8, 1.0, 1.2)
    assert (ev.max_per_class, ev.kid_subsets, ev.kid_class_subset, ev.kid_overall_subset) == (500, 100, 100, 500)
    assert (ev.benchmark_batches, ev.benchmark_warmups, ev.benchmark_repetitions) == ((1, 8), 3, 30)
