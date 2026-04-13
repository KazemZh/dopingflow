from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

log = logging.getLogger(__name__)

_ALLOWED_BACKENDS = {"m3gnet", "uma", "mace", "grace"}

_UMA_MODELS = {"uma-s-1p2", "uma-s-1p1", "uma-m-1p1"}
_UMA_TASKS = {"omat", "oc20", "oc22", "oc25", "omol", "odac", "omc"}

_MACE_MODELS = {
    "small",
    "medium",
    "large",
    "small-mpa-0",
    "medium-mpa-0",
    "large-mpa-0",
    "small-omat-0",
    "medium-omat-0",
}

_GRACE_MODELS = {
    "GRACE-1L-OMAT",
    "GRACE-1L-OMAT-M-base",
    "GRACE-1L-OMAT-M",
    "GRACE-1L-OMAT-L-base",
    "GRACE-1L-OMAT-L",
    "GRACE-2L-OMAT",
    "GRACE-2L-OMAT-M-base",
    "GRACE-2L-OMAT-M",
    "GRACE-2L-OMAT-L-base",
    "GRACE-2L-OMAT-L",
    "GRACE-1L-OAM",
    "GRACE-1L-OAM-M",
    "GRACE-1L-OAM-L",
    "GRACE-2L-OAM",
    "GRACE-2L-OAM-M",
    "GRACE-2L-OAM-L",
    "GRACE-1L-SMAX-L",
    "GRACE-1L-SMAX-OMAT-L",
    "GRACE-2L-SMAX-M",
    "GRACE-2L-SMAX-L",
    "GRACE-2L-SMAX-OMAT-M",
    "GRACE-2L-SMAX-OMAT-L",
}


def set_default_runtime_env(
    *,
    tf_threads: int = 1,
    omp_threads: int = 1,
) -> None:
    """
    Conservative defaults to keep CPU/TensorFlow noise low.
    Safe to call repeatedly.
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("OMP_NUM_THREADS", str(omp_threads))
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(tf_threads))
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", str(tf_threads))


def normalize_backend_config(
    *,
    backend: str,
    model: str,
    task: str,
    section_name: str,
) -> Tuple[str, str, str]:
    """
    Normalize + validate backend/model/task choices for any stage.
    Returns: (backend, model, task)
    """
    backend = str(backend).strip().lower()
    model = str(model).strip()
    task = str(task).strip()

    if backend not in _ALLOWED_BACKENDS:
        raise ValueError(
            f"[{section_name}].backend must be one of: "
            f"{', '.join(sorted(_ALLOWED_BACKENDS))}"
        )

    if backend == "m3gnet":
        if not model:
            model = "default"
        task = ""
        return backend, model, task

    if backend == "uma":
        if model in {"", "default"}:
            model = "uma-s-1p2"
        if task == "":
            task = "omat"

        if model not in _UMA_MODELS:
            raise ValueError(
                f"[{section_name}].model must be one of {sorted(_UMA_MODELS)} "
                "for backend='uma'"
            )
        if task not in _UMA_TASKS:
            raise ValueError(
                f"[{section_name}].task must be one of {sorted(_UMA_TASKS)} "
                "for backend='uma'"
            )
        return backend, model, task

    if backend == "mace":
        if model in {"", "default"}:
            model = "small"
        if model not in _MACE_MODELS:
            raise ValueError(
                f"[{section_name}].model must be one of {sorted(_MACE_MODELS)} "
                "for backend='mace'"
            )
        task = ""
        return backend, model, task

    if backend == "grace":
        if model in {"", "default"}:
            model = "GRACE-1L-OMAT"
        if model not in _GRACE_MODELS:
            raise ValueError(
                f"[{section_name}].model must be one of {sorted(_GRACE_MODELS)} "
                "for backend='grace'"
            )
        task = ""
        return backend, model, task

    raise ValueError(f"Unsupported backend: {backend}")


def check_backend_dependency(backend: str, *, stage_name: str) -> None:
    """
    Fail early with clear stage-specific import errors.
    """
    backend = str(backend).strip().lower()

    if backend == "m3gnet":
        try:
            import m3gnet  # noqa: F401
        except ImportError as e:
            raise ImportError(
                f"{stage_name} backend 'm3gnet' was requested, but M3GNet is not installed."
            ) from e
        return

    if backend == "uma":
        try:
            import fairchem  # noqa: F401
        except ImportError as e:
            raise ImportError(
                f"{stage_name} backend 'uma' was requested, but FAIR-Chem is not installed."
            ) from e
        return

    if backend == "mace":
        try:
            import mace  # noqa: F401
        except ImportError as e:
            raise ImportError(
                f"{stage_name} backend 'mace' was requested, but MACE is not installed."
            ) from e
        return

    if backend == "grace":
        try:
            import tensorpotential  # noqa: F401
        except ImportError as e:
            raise ImportError(
                f"{stage_name} backend 'grace' was requested, but "
                "tensorpotential / grace-tensorpotential is not installed."
            ) from e
        return

    raise ValueError(f"Unsupported backend dependency check: {backend}")


def prepare_backend_runtime(
    *,
    backend: str,
    device: str,
    gpu_id: int,
    tf_threads: int = 1,
    omp_threads: int = 1,
) -> None:
    """
    Configure backend runtime environment in the current process.
    Call this before constructing the calculator.
    """
    import warnings

    backend = str(backend).strip().lower()
    device = str(device).strip().lower()

    if device not in {"cpu", "cuda"}:
        raise ValueError('device must be either "cpu" or "cuda"')
    if gpu_id < 0:
        raise ValueError("gpu_id must be >= 0")

    os.environ["OMP_NUM_THREADS"] = str(omp_threads)

    if backend == "m3gnet":
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
        os.environ["TF_NUM_INTRAOP_THREADS"] = str(tf_threads)
        os.environ["TF_NUM_INTEROP_THREADS"] = str(tf_threads)

        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", message=".*experimental_relax_shapes.*")
        warnings.filterwarnings("ignore", message=".*casting an input of type complex64.*")

        logging.getLogger("tensorflow").setLevel(logging.ERROR)

        try:
            import tensorflow as tf

            tf.get_logger().setLevel("ERROR")
            try:
                tf.autograph.set_verbosity(0)
            except Exception:
                pass

            gpus = tf.config.list_physical_devices("GPU")

            if device == "cpu":
                try:
                    tf.config.set_visible_devices([], "GPU")
                except Exception:
                    pass
            else:
                if not gpus:
                    raise RuntimeError(
                        f"CUDA requested for backend='{backend}', but TensorFlow sees no GPU."
                    )
                if gpu_id >= len(gpus):
                    raise RuntimeError(
                        f"Requested gpu_id={gpu_id}, but only {len(gpus)} GPU(s) are visible."
                    )
                try:
                    tf.config.set_visible_devices(gpus[gpu_id], "GPU")
                    tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
                except Exception:
                    pass

        except Exception:
            if device == "cuda":
                raise

        return

    if backend == "uma":
        if device == "cuda":
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        return

    if backend == "mace":
        os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        if device == "cuda":
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        return

    if backend == "grace":
        if device == "cuda":
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        return

    raise ValueError(f"Unsupported backend runtime preparation: {backend}")


def build_ase_calculator(
    *,
    backend: str,
    model: str,
    task: str,
    device: str,
):
    """
    Return an ASE-compatible calculator for the selected backend.
    """
    backend = str(backend).strip().lower()

    if backend == "m3gnet":
        try:
            from m3gnet.models import M3GNet, M3GNetCalculator, Potential
        except ImportError as e:
            raise ImportError(
                "M3GNet backend requested, but m3gnet is not installed."
            ) from e

        model_obj = M3GNet.load()
        potential = Potential(model_obj)
        return M3GNetCalculator(potential=potential)

    if backend == "uma":
        try:
            from fairchem.core import FAIRChemCalculator, pretrained_mlip
        except ImportError as e:
            raise ImportError(
                "UMA backend requested, but FAIR-Chem is not installed."
            ) from e

        predictor = pretrained_mlip.get_predict_unit(model, device=device)
        return FAIRChemCalculator(predictor, task_name=task)

    if backend == "mace":
        try:
            from mace.calculators import mace_mp
        except ImportError as e:
            raise ImportError(
                "MACE backend requested, but mace-torch is not installed."
            ) from e

        return mace_mp(model=model, device=device, default_dtype="float64")

    if backend == "grace":
        try:
            from tensorpotential.calculator.foundation_models import grace_fm
        except ImportError as e:
            raise ImportError(
                "GRACE backend requested, but tensorpotential / grace-tensorpotential "
                "is not installed."
            ) from e

        return grace_fm(model)

    raise ValueError(f"Unsupported backend calculator build: {backend}")