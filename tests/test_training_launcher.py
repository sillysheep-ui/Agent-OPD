from pathlib import Path


def test_training_launcher_forces_fsdp1_after_caller_overrides():
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts" / "run_verl_train.sh").read_text(encoding="utf-8")
    trainer_command = source.split('"${trainer_path}" \\', 1)[1].split('2>&1 | tee', 1)[0]
    assert trainer_command.index('"$@" \\') < trainer_command.index('model.strategy=fsdp \\')


def test_training_launcher_puts_hydra_output_with_run_artifacts():
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts" / "run_verl_train.sh").read_text(encoding="utf-8")
    trainer_command = source.split('"${trainer_path}" \\', 1)[1].split('2>&1 | tee', 1)[0]
    assert trainer_command.index('"$@" \\') < trainer_command.index(
        'hydra.run.dir="${OUTPUT_DIR}/hydra" \\'
    )
