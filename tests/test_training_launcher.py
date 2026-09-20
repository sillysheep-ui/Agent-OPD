from pathlib import Path

import yaml


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


def test_training_launcher_uses_repo_owned_v080_config_and_release_tag():
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts" / "run_verl_train.sh").read_text(encoding="utf-8")
    assert 'verl_config_dir="${repo_root}/configs"' in source
    assert '--config-name verl_v080_omniopd_sft \\' in source
    assert 'required_verl_version = "0.8.0.dev"' in source
    assert 'if release_tag != "v0.8.0":' in source
    assert '"verl_config": Path(os.environ["OMNIOPD_VERL_CONFIG_PATH"]).resolve()' in source


def test_repo_owned_v080_config_has_custom_trainer_contract():
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root / "configs" / "verl_v080_omniopd_sft.yaml").read_text(encoding="utf-8")
    )
    assert config["model"]["strategy"] == "fsdp"
    assert config["model"]["fsdp_config"]["model_dtype"] == "fp32"
    assert config["data"]["micro_batch_size_per_gpu"] == 1
    assert config["use_remove_padding"] is False
    assert config["ulysses_sequence_parallel_size"] == 1
