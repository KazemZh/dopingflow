"""Slurm GPU visibility regression tests for MACE runtime."""

from dopingflow.ml_backends import prepare_backend_runtime


def test_mace_respects_scheduler_gpu_mask(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    prepare_backend_runtime(
        backend="mace", device="cuda", gpu_id=0, tf_threads=1, omp_threads=8
    )
    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == "2"


def test_mace_sets_gpu_mask_outside_scheduler(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    prepare_backend_runtime(
        backend="mace", device="cuda", gpu_id=0, tf_threads=1, omp_threads=8
    )
    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == "0"
