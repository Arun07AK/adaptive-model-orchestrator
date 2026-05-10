import pytest

from scripts.quick_bench import ALL_CONFIGS, BENCHMARK_CONFIGS, expand_configs


def test_all_config_expands_to_every_supported_benchmark_config():
    assert expand_configs("all") == list(BENCHMARK_CONFIGS)
    assert ALL_CONFIGS == BENCHMARK_CONFIGS
    assert "qwen235b_standalone" in ALL_CONFIGS
    assert "orchestrated" in ALL_CONFIGS
    assert "hybrid" in ALL_CONFIGS
    assert "v3_cross_model" in ALL_CONFIGS


def test_single_config_expands_to_itself():
    assert expand_configs("cascade") == ["cascade"]


def test_unknown_config_is_rejected():
    with pytest.raises(ValueError, match="Unknown config"):
        expand_configs("does_not_exist")
