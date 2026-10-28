"""No-torch static acceptance checks for Phase 3 supplementary runners."""
from pathlib import Path
from scripts import run_cls_lora_supplementary as runner


def test_plans_preserve_formal_cls_lora_and_seal_test(tmp_path):
    multi = runner.parse_args(["multiseed"])
    assert runner.seeds(multi) == (123, 2026)
    command = runner.build_command(multi, 123, tmp_path / "seed_123")
    assert "--evaluate-test" not in command and "--comparison-mode" in command
    for flag, value in (("--pooling", "cls"), ("--training-strategy", "lora"), ("--adapter-learning-rate", "5e-5"), ("--seed", "123")):
        assert command[command.index(flag) + 1] == value
    ablation = runner.parse_args(["unweighted-ablation"])
    command = runner.build_command(ablation, 42, tmp_path / "seed_42")
    assert runner.seeds(ablation) == (42,)
    assert command[command.index("--class-weighting") + 1] == "unweighted"
    assert "--comparison-mode" not in command


def test_runner_rejects_test_style_options():
    try:
        runner.parse_args(["multiseed", "--evaluate-test"])
    except SystemExit:
        return
    raise AssertionError("Test option must be rejected")
