import ast
from pathlib import Path
from typing import List, Optional, Tuple

import click
import numpy as np
import pandas as pd
import torch

from pairwise_bo.types import GenericCandidate

def save_ranking_test_set(
    items: List[GenericCandidate],
    utilities: List[float],
    output_path: Path,
) -> None:
    if not items or not utilities or len(items) != len(utilities):
        click.echo("Invalid ranking test set; nothing to save.", err=True)
        return
    data = [
        {"item": it.to_dict(), "utility": float(u)} for it, u in zip(items, utilities)
    ]
    df = pd.DataFrame(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    click.echo(f"Ranking test set saved to {output_path}")


def load_ranking_test_set(
    test_set_path: Path,
    feature_mapping: dict,
) -> Tuple[List[GenericCandidate], List[float]]:
    if not test_set_path.exists():
        click.echo(
            f"Ranking test set file not found at {test_set_path}",
            err=True,
        )
        return [], []
    df = pd.read_csv(test_set_path)
    items: List[GenericCandidate] = []
    utilities: List[float] = []
    for _, row in df.iterrows():
        items.append(
            GenericCandidate(
                feature_keys=list(feature_mapping.keys()),
                values=[float(ast.literal_eval(row["item"])[k]) for k in feature_mapping.keys()]
            )
        )
        utilities.append(float(row["utility"]))
    click.echo(f"Loaded ranking test set with {len(items)} items from {test_set_path}")
    return items, utilities


def _rank_from_utilities(utilities: List[float]) -> List[int]:
    util = np.asarray(utilities, dtype=float)
    order = np.lexsort((np.arange(len(util)), -util))
    return order.tolist()


def _pairwise_ranking_accuracy(pred_rank: List[int], gt_rank: List[int]) -> float:
    n = len(gt_rank)
    if n < 2:
        return 1.0
    pos_pred = {item: i for i, item in enumerate(pred_rank)}
    pos_gt = {item: i for i, item in enumerate(gt_rank)}
    total = n * (n - 1) // 2
    correct = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = gt_rank[i], gt_rank[j]
            correct += int((pos_pred[a] < pos_pred[b]) == (pos_gt[a] < pos_gt[b]))
    return correct / total


def _spearman_rho(pred_rank: List[int], gt_rank: List[int]) -> float:
    n = len(gt_rank)
    if n < 2:
        return 1.0
    pos_pred = {item: i for i, item in enumerate(pred_rank)}
    pos_gt = {item: i for i, item in enumerate(gt_rank)}
    d2 = 0.0
    for item in gt_rank:
        d = pos_pred[item] - pos_gt[item]
        d2 += d * d
    return 1.0 - (6.0 * d2) / (n * (n * n - 1.0))


def _ndcg(
    pred_rank: List[int],
    gt_utilities: List[float],
    k: Optional[int] = None,
    n_bins: int = 5,
) -> float:
    n = len(gt_utilities)
    if n == 0:
        return 0.0
    if k is None or k > n:
        k = n

    if n < n_bins:
        util = np.asarray(gt_utilities, dtype=float)
        umin, umax = float(util.min()), float(util.max())
        rel = (util - umin) / (umax - umin) if umax > umin else np.zeros_like(util)
    else:
        try:
            rel_discrete = pd.qcut(
                gt_utilities, q=n_bins, labels=False, duplicates="drop"
            )
            rel = np.asarray(rel_discrete, dtype=float)
        except ValueError:
            util = np.asarray(gt_utilities, dtype=float)
            umin, umax = float(util.min()), float(util.max())
            rel = (util - umin) / (umax - umin) if umax > umin else np.zeros_like(util)

    def dcg(order: List[int], kk: int) -> float:
        s = 0.0
        for rank_pos in range(kk):
            idx = order[rank_pos]
            s += rel[idx] / np.log2(rank_pos + 2.0)
        return s

    ideal_order = _rank_from_utilities(gt_utilities)
    idcg = dcg(ideal_order, k)
    return (dcg(pred_rank, k) / idcg) if idcg > 0 else 0.0


def _top1_accuracy(pred_rank: List[int], gt_rank: List[int]) -> float:
    return 1.0 if (len(gt_rank) > 0 and pred_rank[0] == gt_rank[0]) else 0.0


def calculate_ranking_test_set_metrics(
    elicitator: Any,
    items: List[GenericCandidate],
    utilities: List[float],
) -> dict[str, float]:
    if not items or not utilities or len(items) != len(utilities):
        return {
            "pairwise_accuracy": 0.0,
            "spearman_rho": 0.0,
            "ndcg_1": 0.0,
            "ndcg_3": 0.0,
            "ndcg_5": 0.0,
            "ndcg_10": 0.0,
            "ndcg_full": 0.0,
            "top1_accuracy": 0.0,
        }

    X = torch.tensor(np.vstack([c.to_numpy() for c in items]), dtype=torch.float32)
    sorted_indices, _scores = elicitator.rank_listings(X, return_scores=True)
    pred_order = [
        int(i)
        for i in (
            sorted_indices.tolist()
            if hasattr(sorted_indices, "tolist")
            else list(sorted_indices)
        )
    ]

    gt_order = _rank_from_utilities(utilities)

    pacc = _pairwise_ranking_accuracy(pred_order, gt_order)
    rho = _spearman_rho(pred_order, gt_order)
    ndcg_1 = _ndcg(pred_order, utilities, k=1)
    ndcg_3 = _ndcg(pred_order, utilities, k=3)
    ndcg_5 = _ndcg(pred_order, utilities, k=5)
    ndcg_10 = _ndcg(pred_order, utilities, k=10)
    ndcg_full = _ndcg(pred_order, utilities)
    top1 = _top1_accuracy(pred_order, gt_order)

    return {
        "pairwise_accuracy": pacc,
        "spearman_rho": rho,
        "ndcg_1": ndcg_1,
        "ndcg_3": ndcg_3,
        "ndcg_5": ndcg_5,
        "ndcg_10": ndcg_10,
        "ndcg_full": ndcg_full,
        "top1_accuracy": top1,
    }


def create_ranking_test_set_with_profile(
    dataset: torch.Tensor,
    reverse_scaling_fn: Any,
    n_items: int,
    profile_weights: torch.Tensor,
    feature_mapping: dict,
) -> Tuple[List[GenericCandidate], List[float]]:
    n_total = dataset.shape[0]
    n_items = min(n_items, n_total)
    idx = torch.randperm(n_total)[:n_items].tolist()
    items = [
        GenericCandidate.from_numpy(
            dataset[i].numpy(), list(feature_mapping.keys())
        )
        for i in idx
    ]
    
    utilities = []
    for it in items:
        # Re-scale back to original values before multiplying with profile weights
        unscaled_array = reverse_scaling_fn(it.to_numpy().reshape(1, -1)).squeeze(0)
        unscaled_tensor = torch.tensor(unscaled_array, dtype=torch.float32)
        utility = (profile_weights * unscaled_tensor).sum().item()
        utilities.append(utility)

    indices = torch.randperm(len(items)).tolist()
    items = [items[i] for i in indices]
    utilities = [utilities[i] for i in indices]
    return items, utilities


def get_profile_utility(profile_weights: torch.Tensor, candidate: GenericCandidate) -> float:
    return float(torch.dot(profile_weights, torch.tensor(candidate.values, dtype=torch.float32)).item())


def get_profile_preference(
    candidate_pair: Any, profile_weights: torch.Tensor
) -> int:
    ua = get_profile_utility(profile_weights, candidate_pair.listing_a)
    ub = get_profile_utility(profile_weights, candidate_pair.listing_b)
    prob_a = 1.0 / (1.0 + np.exp(-(ua - ub)))
    if prob_a > 0.5:
        return 0
    if prob_a < 0.5:
        return 1
    return 0
