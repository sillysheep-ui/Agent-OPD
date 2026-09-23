"""Contracts for turning a veRL FSDP LoRA shard into a PEFT adapter.

The 2026-09-23 run exposed two silent traps: shard files must be ordered
numerically (rank 10 sorts before rank 2 as text), and every LoRA tensor arrives
as a DTensor whose local piece must be read per rank and concatenated on the
placement dimension. These helpers are pure so the contract is testable without
torch or a GPU.
"""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "convert_opd_checkpoint", ROOT / "scripts" / "convert_opd_checkpoint.py"
)
convert = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(convert)


class _Shard:
    def __init__(self, dim):
        self.dim = dim


class _Replicate:
    pass


class _DTensor:
    def __init__(self, placements):
        self.placements = placements


def test_shard_files_are_ordered_numerically():
    paths = [
        Path("model_world_size_2_rank_10.pt"),
        Path("model_world_size_2_rank_2.pt"),
        Path("model_world_size_2_rank_1.pt"),
    ]
    assert [p.name for p in sorted(paths, key=convert.rank_sort_key)] == [
        "model_world_size_2_rank_1.pt",
        "model_world_size_2_rank_2.pt",
        "model_world_size_2_rank_10.pt",
    ]
    assert convert.rank_sort_key(Path("model_world_size_4_rank_3.pt")) == (4, 3)
    try:
        convert.rank_sort_key(Path("model.pt"))
    except ValueError:
        pass
    else:
        raise AssertionError("a shard name without world size/rank was accepted")


def test_peft_name_maps_only_lora_pairs():
    base = "base_model.model.model.layers.0.self_attn.q_proj"
    assert (
        convert.peft_name(f"_fsdp_wrapped_module.{base}.lora_A.default.weight")
        == f"{base}.lora_A.weight"
    )
    assert (
        convert.peft_name(f"{base}.lora_B.default.weight") == f"{base}.lora_B.weight"
    )
    assert convert.peft_name(f"{base}.base_layer.weight") is None


def test_placement_dim_reads_shard_and_replicate():
    assert convert.placement_dim(_DTensor([_Shard(0)])) == 0
    assert convert.placement_dim(_DTensor([_Shard(1)])) == 1
    assert convert.placement_dim(_DTensor([_Replicate()])) is None
    assert convert.placement_dim("plain tensor") is None


def test_concatenated_shape_rebuilds_the_global_shape():
    assert convert.concatenated_shape([(8, 2560), (8, 2560)], 0) == (16, 2560)
    assert convert.concatenated_shape([(2560, 8), (2560, 8)], 1) == (2560, 16)
    # Sharding an axis unevenly is legal; disagreeing on any other axis is not.
    assert convert.concatenated_shape([(8, 2560), (4, 2560)], 0) == (12, 2560)
    for bad, dim in (
        ([(8, 2560), (8, 1280)], 0),
        ([(2560, 8), (1280, 8)], 1),
    ):
        try:
            convert.concatenated_shape(bad, dim)
        except ValueError:
            continue
        raise AssertionError(f"inconsistent shards were accepted: {bad} on dim {dim}")
    try:
        convert.concatenated_shape([(8, 2560)], 2)
    except ValueError:
        pass
    else:
        raise AssertionError("an out-of-range shard dimension was accepted")


def test_base_weight_key_maps_adapter_keys_onto_base_layers():
    assert (
        convert.base_weight_key(
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
        )
        == "model.layers.0.self_attn.q_proj.weight"
    )
    assert (
        convert.base_weight_key("base_model.model.model.layers.3.mlp.up_proj.lora_B.weight")
        == "model.layers.3.mlp.up_proj.weight"
    )
    try:
        convert.base_weight_key("model.layers.0.self_attn.q_proj.lora_A.weight")
    except ValueError:
        pass
    else:
        raise AssertionError("an adapter key without a base prefix was accepted")
