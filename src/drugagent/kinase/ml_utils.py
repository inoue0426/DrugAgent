from __future__ import annotations

import contextlib
import io
import math
import os
from typing import List, Sequence, Tuple, Union

import numpy as np

from .ml_cache_utils import (
    ML_LOOKUP_CACHE_DIR,
    ML_SCORE_CACHE_DIR,
    get_cached_sequence,
    get_cached_smiles,
    init_cache,
)

# Define a type alias for clarity
ResultType = List[Union[str, float]]

# Global cached pretrained net
_PRETRAINED_NET = None


def _prepare_ml_runtime() -> None:
    """Configure runtime settings to reduce kernel crashes.

    This forces CPU-only execution and limits thread usage to avoid
    native library instability in notebook kernels.
    """
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        # If torch is unavailable or fails to configure, continue.
        pass


def _normalize_inputs(
    drug_names: Union[str, Sequence[str]],
    target_names: Union[str, Sequence[str]],
) -> Tuple[List[str], List[str]]:
    """Normalize inputs to lists of strings for scoring.

    Args:
        drug_names: Drug name or list of drug names.
        target_names: Target name or list of target names.

    Returns:
        Tuple of (drug_names, target_names) as lists.
    """
    if isinstance(drug_names, str):
        drug_list = [drug_names]
    else:
        drug_list = list(drug_names)

    if isinstance(target_names, str):
        target_list = [target_names]
    else:
        target_list = list(target_names)

    if len(drug_list) != len(target_list):
        raise ValueError(
            "drug_names and target_names must have the same length "
            f"(got {len(drug_list)} and {len(target_list)})"
        )

    return drug_list, target_list


def _get_pretrained_bindingdb_model(model_name: str = "Transformer_CNN_BindingDB"):
    """Load and cache pretrained DeepPurpose BindingDB model (single global instance)."""
    global _PRETRAINED_NET
    if _PRETRAINED_NET is None:
        from DeepPurpose import DTI as models

        print(f"[ml_utils] Loading DeepPurpose model: {model_name}")
        # This will download if not available locally.
        _PRETRAINED_NET = models.model_pretrained(model=model_name)
    return _PRETRAINED_NET


def pKd_to_KdM(pkd: float) -> float:
    """Convert pKd to Kd in molar units."""
    return 10 ** (-float(pkd))


def pKd_to_class(
    pkd: float,
    t_weak: float = 6.0,
    t_strong: float = 7.0,
    binary: bool = False,
) -> str:
    """Map pKd to class labels.

    Args:
        pkd: pKd value.
        t_weak: Threshold for Weak vs Moderate.
        t_strong: Threshold for Moderate vs Strong.
        binary: If True, return Active/Inactive labels (Strong only is Active).

    Returns:
        Class label string.
    """
    if math.isnan(pkd):
        return "NA"
    if binary:
        return "Active" if pkd >= t_strong else "Inactive"
    if pkd >= t_strong:
        return "Strong"
    if pkd >= t_weak:
        return "Moderate"
    return "Weak"


def get_dti_score(
    drug_names: Union[str, Sequence[str]],
    target_names: Union[str, Sequence[str]],
    model_name: str = "Transformer_CNN_BindingDB",
    t_weak: float = 6.0,
    t_strong: float = 7.0,
    binary_class: bool = False,
) -> List[ResultType]:
    """Score drug-target interactions using a pretrained DeepPurpose BindingDB model.

    Args:
        drug_names: Drug name or list of drug names.
        target_names: Target name or list of target names.
        model_name: Name of pretrained DeepPurpose model to load.
        t_weak/t_strong: thresholds (on pKd) for Weak/Moderate/Strong.
        binary_class: If True, return Active/Inactive (Strong only is Active).

    Returns:
        A list of records:
            [drug_name, target_name, pKd (float), class_label (str),
             Kd_M (float or None), Kd_nM (float or None), reasoning (str)]
    """
    _prepare_ml_runtime()
    from DeepPurpose import DTI as models

    drug_list, target_list = _normalize_inputs(drug_names, target_names)
    reasoning_base = "This agent used DeepPurpose pretrained BindingDB model"
    result: List[Optional[ResultType]] = [None] * len(drug_list)

    score_cache = init_cache(ML_SCORE_CACHE_DIR)
    lookup_cache = init_cache(ML_LOOKUP_CACHE_DIR)

    try:
        uncached_pairs = []

        # 1) Cache check + lookups
        for i, (drug_name, target_name) in enumerate(zip(drug_list, target_list)):
            cache_key = f"{drug_name}_{target_name}"

            if cache_key in score_cache:
                try:
                    pKd_cached = float(score_cache[cache_key])
                except Exception:
                    pKd_cached = float("nan")
                cls = pKd_to_class(
                    pKd_cached, t_weak=t_weak, t_strong=t_strong, binary=binary_class
                )
                kdM = pKd_to_KdM(pKd_cached) if not math.isnan(pKd_cached) else None
                kd_nM = kdM * 1e9 if kdM is not None else None
                result[i] = [
                    drug_name,
                    target_name,
                    pKd_cached,
                    cls,
                    kdM,
                    kd_nM,
                    reasoning_base + " (cached)",
                ]
                continue

            drug = get_cached_smiles(lookup_cache, drug_name)
            target = get_cached_sequence(lookup_cache, target_name)

            if not drug or not target:
                result[i] = [
                    drug_name,
                    target_name,
                    float("nan"),
                    "Missing_SMILES_or_sequence",
                    None,
                    None,
                    "Missing SMILES or sequence for ML scoring",
                ]
                continue

            uncached_pairs.append((i, drug, target, drug_name, target_name))

        # 2) Process uncached pairs using pretrained model
        if uncached_pairs:
            net = _get_pretrained_bindingdb_model(model_name=model_name)

            indices = [p[0] for p in uncached_pairs]
            drugs = [p[1] for p in uncached_pairs]
            targets = [p[2] for p in uncached_pairs]
            drug_names_uncached = [p[3] for p in uncached_pairs]
            target_names_uncached = [p[4] for p in uncached_pairs]

            # suppress DeepPurpose stdout/stderr
            try:
                with (
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    preds = models.virtual_screening(
                        drugs,
                        targets,
                        net,
                        drug_names_uncached,
                        target_names_uncached,
                    )
            except Exception as e:
                raise RuntimeError(f"DeepPurpose virtual_screening failed: {e}") from e

            arr = np.asarray(preds)
            # normalize to 1D array of pKd floats
            if arr.ndim == 0:
                arr = arr.reshape(1)
            if arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr.reshape(-1)
            elif arr.ndim > 1:
                # if multi-d output, take first column
                arr = arr[:, 0]

            if arr.shape[0] != len(indices):
                raise RuntimeError(
                    f"Prediction count mismatch: expected {len(indices)}, got {arr.shape[0]}"
                )

            for idx, dname, tname, raw_p in zip(
                indices, drug_names_uncached, target_names_uncached, arr
            ):
                try:
                    pKd = float(raw_p)
                except Exception:
                    pKd = float("nan")

                # sanitize NaN/Inf
                if math.isnan(pKd) or math.isinf(pKd):
                    pKd = float("nan")

                cache_key = f"{dname}_{tname}"
                # store pKd in cache (so thresholding can be changed later)
                score_cache[cache_key] = pKd

                cls = (
                    pKd_to_class(
                        pKd, t_weak=t_weak, t_strong=t_strong, binary=binary_class
                    )
                    if not math.isnan(pKd)
                    else "NA"
                )
                kdM = pKd_to_KdM(pKd) if not math.isnan(pKd) else None
                kd_nM = kdM * 1e9 if kdM is not None else None

                result[idx] = [
                    dname,
                    tname,
                    pKd,
                    cls,
                    kdM,
                    kd_nM,
                    reasoning_base,
                ]
    finally:
        try:
            score_cache.close()
        except Exception:
            pass
        try:
            lookup_cache.close()
        except Exception:
            pass

    # Filter out any None and return
    return [r for r in result if r is not None]
