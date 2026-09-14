from __future__ import annotations
import math
from typing import Dict, List
import numpy as np

def finite(value: float) -> bool:
    return math.isfinite(float(value))



def adjust_fdr(rows: List[Dict], pvalue_key: str, output_key: str) -> None:
    valid = sorted(
        (
            (index, float(row[pvalue_key]))
            for index, row in enumerate(rows)
            if finite(row[pvalue_key])
        ),
        key=lambda item: item[1],
    )
    running = 1.0
    adjusted = [1.0] * len(valid)
    for rank in range(len(valid), 0, -1):
        running = min(running, valid[rank - 1][1] * len(valid) / rank)
        adjusted[rank - 1] = running
    for position, (row_index, _) in enumerate(valid):
        rows[row_index][output_key] = adjusted[position]
    for row in rows:
        row.setdefault(output_key, float("nan"))



def standardize_geometry(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = raw.mean(axis=0)
    scale = raw.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return (raw - mean) / scale, mean, scale



def build_design(
    geometry: np.ndarray,
    origin_quality: np.ndarray,
    identities: np.ndarray,
    family_ids: List[str],
) -> tuple[np.ndarray, List[str], int]:
    geometry_z, _, _ = standardize_geometry(geometry)
    origin_q = origin_quality.reshape(-1, 1)
    origin_q = (origin_q - origin_q.mean(axis=0)) / max(float(origin_q.std()), 1e-8)
    nuisance = np.concatenate([geometry_z, geometry_z**2, origin_q], axis=1)
    names = (
        ["intercept"]
        + [f"geometry_{index}" for index in range(geometry_z.shape[1])]
        + [f"geometry_squared_{index}" for index in range(geometry_z.shape[1])]
        + ["origin_q50"]
        + [f"family:{family_id}" for family_id in family_ids]
    )
    family_start = 1 + nuisance.shape[1]
    design = np.concatenate(
        [np.ones((len(geometry), 1)), nuisance, identities.astype(np.float64)], axis=1
    )
    return design, names, family_start



def penalty_matrix(columns: int, family_start: int, alpha: float, rows: int) -> np.ndarray:
    penalty = np.eye(columns, dtype=np.float64) * (1e-10 * max(rows, 1))
    penalty[:family_start, :family_start] = 0.0
    if alpha > 0:
        positions = np.arange(family_start, columns)
        penalty[positions, positions] = alpha * max(rows, 1)
    return penalty



def fit_ridge(
    design: np.ndarray, outcome: np.ndarray, family_start: int, alpha: float
) -> np.ndarray:
    lhs = design.T @ design + penalty_matrix(
        design.shape[1], family_start, alpha, len(design)
    )
    rhs = design.T @ outcome
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(lhs) @ rhs



def clustered_bootstrap_coefficients(
    design: np.ndarray,
    outcome: np.ndarray,
    scenes: List[str],
    family_start: int,
    alpha: float,
    samples: int,
    chunk: int,
    seed: int,
) -> np.ndarray:
    unique = {scene: index for index, scene in enumerate(sorted(set(scenes)))}
    inverse = np.asarray([unique[scene] for scene in scenes], dtype=np.int64)
    groups = len(unique)
    columns = design.shape[1]
    scene_xtx = np.zeros((groups, columns, columns), dtype=np.float64)
    scene_xty = np.zeros((groups, columns), dtype=np.float64)
    scene_sizes = np.zeros(groups, dtype=np.float64)
    for group in range(groups):
        mask = inverse == group
        local = design[mask]
        scene_xtx[group] = local.T @ local
        scene_xty[group] = local.T @ outcome[mask]
        scene_sizes[group] = int(mask.sum())

    rng = np.random.default_rng(seed)
    coefficients = []
    probability = np.full(groups, 1.0 / groups)
    family_positions = np.arange(family_start, columns)
    all_positions = np.arange(columns)
    for start in range(0, samples, chunk):
        count = min(chunk, samples - start)
        weights = rng.multinomial(groups, probability, size=count).astype(np.float64)
        lhs = np.einsum("bg,gij->bij", weights, scene_xtx, optimize=True)
        rhs = weights @ scene_xty
        sampled_rows = weights @ scene_sizes
        lhs[:, all_positions, all_positions] += 1e-10 * sampled_rows[:, None]
        if alpha > 0:
            lhs[:, family_positions, family_positions] += (
                alpha * sampled_rows[:, None]
            )
        try:
            beta = np.linalg.solve(lhs, rhs[..., None]).squeeze(-1)
        except np.linalg.LinAlgError:
            beta = np.stack(
                [np.linalg.pinv(matrix) @ vector for matrix, vector in zip(lhs, rhs)]
            )
        coefficients.append(beta[:, family_start:])
    return np.concatenate(coefficients, axis=0)



def centered_bootstrap_pvalue(
    bootstrap: np.ndarray, observed: float, threshold: float
) -> float:
    centered = bootstrap - observed
    distance = observed - threshold
    return (int((centered >= distance).sum()) + 1) / (len(centered) + 1)



def nanmean(matrix: np.ndarray, valid: np.ndarray, axis: int) -> np.ndarray:
    numerator = np.where(valid, matrix, 0.0).sum(axis=axis)
    denominator = valid.sum(axis=axis)
    output = np.full_like(numerator, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=output, where=denominator > 0)
    return output



def leave_one_style_out(
    design: np.ndarray,
    centered_q: np.ndarray,
    valid: np.ndarray,
    family_start: int,
    alpha: float,
) -> np.ndarray:
    estimates = []
    for style in range(centered_q.shape[1]):
        keep = valid.copy()
        keep[:, style] = False
        outcome = nanmean(centered_q, keep, axis=1)
        row_valid = np.isfinite(outcome)
        beta = fit_ridge(
            design[row_valid], outcome[row_valid], family_start, alpha
        )
        estimates.append(beta[family_start:])
    return np.stack(estimates, axis=0)

