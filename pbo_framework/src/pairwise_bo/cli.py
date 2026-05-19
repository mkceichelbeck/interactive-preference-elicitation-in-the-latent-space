"""CLI for pairwise BO workflows."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import click
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from pairwise_bo.factory import build_loader, get_elicitator
from pairwise_bo.llm_client import build_llm_client
from pairwise_bo.llm_prompts import (
    LLMUsageTracker,
    get_user_weights_and_bounds,
    get_llm_preference,
)
from pairwise_bo.profiles import load_profile_weights
from pairwise_bo.evaluation import (
    create_ranking_test_set_with_profile,
    calculate_ranking_test_set_metrics,
    get_profile_preference,
)


def _parse_weights(
    weights: Optional[str],
    dim: int,
    seed: int,
) -> torch.Tensor:
    if weights is not None and weights.strip():
        values = [float(v.strip()) for v in weights.split(",") if v.strip()]
        if len(values) != dim:
            raise click.UsageError(
                f"Expected {dim} weight values, but got {len(values)}."
            )
        tensor = torch.tensor(values, dtype=torch.float32)
        if torch.all(tensor == 0):
            raise click.UsageError("At least one weight must be non-zero.")
        return tensor

    generator = torch.Generator().manual_seed(seed)
    sampled = torch.randn(dim, generator=generator)
    if torch.all(sampled == 0):
        sampled[0] = 1.0
    return sampled


@click.group()
def cli() -> None:
    """Pairwise BO CLI."""


@cli.command("smoke")
@click.option(
    "--data-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to CSV dataset.",
)
@click.option(
    "--dataset-preset",
    type=click.Choice(["idealista-madrid", "idealista", "munich"], case_sensitive=False),
    default="idealista-madrid",
    show_default=True,
    help="Feature schema preset to apply.",
)
@click.option(
    "--scaler-type",
    type=click.Choice(["none", "minmax", "standard", "robust"], case_sensitive=False),
    default="robust",
    show_default=True,
    help="Feature scaler used during loading.",
)
@click.option(
    "--weights",
    type=str,
    default=None,
    help="Comma-separated utility weights ordered by active feature schema.",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    show_default=True,
    help="Global random seed for reproducible smoke runs.",
)
@click.option(
    "--bo-model-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional path to a saved BO model state.",
)
@click.option(
    "--ae-checkpoint",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional path to a vanilla autoencoder checkpoint.",
)
def smoke(
    data_path: Path,
    dataset_preset: str,
    scaler_type: str,
    weights: Optional[str],
    seed: int,
    bo_model_path: Optional[Path],
    ae_checkpoint: Optional[Path],
) -> None:
    """Run a deterministic end-to-end BO smoke iteration."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    loader = build_loader(
        data_path=data_path,
        dataset_preset=dataset_preset,
        scaler_type=scaler_type,
    )
    user_weights = _parse_weights(weights=weights, dim=loader.dim, seed=seed)

    elicitator = get_elicitator(
        data_path=data_path,
        bo_model_path=bo_model_path,
        autoencoder_model_path=ae_checkpoint,
        user_feature_weights=user_weights,
        dataset_preset=dataset_preset,
        scaler_type=scaler_type,
    )

    candidate_pair = elicitator.select_next_candidate_pair()
    predicted_response = elicitator.predict_choice(candidate_pair.to_tensor())
    elicitator.handle_user_response(candidate_pair, predicted_response)

    sample_count = min(20, int(elicitator.data.shape[0]))
    subset = elicitator.data[:sample_count]
    ranked_indices, ranked_scores = elicitator.rank_listings(
        subset,
        return_scores=True,
    )

    click.echo(
        "SMOKE_OK "
        f"comparisons={elicitator.total_compare_count} "
        f"top_index={int(ranked_indices[0])} "
        f"top_score={float(ranked_scores[0]):.6f}"
    )


@cli.command("llm-eval")
@click.option(
    "--data-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to CSV dataset.",
)
@click.option(
    "--dataset-preset",
    type=click.Choice(["idealista-madrid", "idealista", "munich"], case_sensitive=False),
    default="idealista-madrid",
    show_default=True,
    help="Feature schema preset to apply.",
)
@click.option(
    "--scaler-type",
    type=click.Choice(["none", "minmax", "standard", "robust"], case_sensitive=False),
    default="robust",
    show_default=True,
    help="Feature scaler used during loading.",
)
@click.option(
    "--api-key",
    type=str,
    default=None,
    help="API Key for the LLM. If not provided, falls back to GEMINI_API_KEY env variable.",
)
@click.option(
    "--model",
    type=str,
    default="gemini:gemini-2.5-flash",
    show_default=True,
    help="LLM model string (e.g., gemini:gemini-2.5-flash or ollama:llama3).",
)
@click.option(
    "--persona",
    type=str,
    default="budget_conscious",
    help="Persona for the LLM to adopt (e.g. family_friendly, budget_conscious).",
)
@click.option(
    "--num-loops",
    type=int,
    default=15,
    show_default=True,
    help="Number of active learning loops.",
)
@click.option(
    "--num-test-items",
    type=int,
    default=100,
    show_default=True,
    help="Size of ranking test set.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default="evaluation_results",
    help="Directory to save metric traces.",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    show_default=True,
    help="Global random seed.",
)
def llm_eval(
    data_path: Path,
    dataset_preset: str,
    scaler_type: str,
    api_key: Optional[str],
    model: str,
    persona: str,
    num_loops: int,
    num_test_items: int,
    seed: int,
) -> None:
    """Run a full evaluation loop using an LLM for preference elicitation."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    llm_client = build_llm_client(model_name=model, api_key=api_key)
    usage_tracker = LLMUsageTracker(model_name=model)

    loader = build_loader(
        data_path=data_path,
        dataset_preset=dataset_preset,
        scaler_type=scaler_type,
    )
    feature_mapping = loader.feature_mapping

    gt_weights, _ = load_profile_weights(persona, dataset_preset, feature_mapping)
    
    click.echo("Generating ranking test set based on ground-truth profile weights...")
    test_items, test_utilities = create_ranking_test_set_with_profile(
        loader.data,
        loader.scaler.inverse_transform if hasattr(loader, "scaler") and loader.scaler else lambda x: x,
        num_test_items,
        gt_weights,
        feature_mapping,
    )

    click.echo(f"Initialized LLM ({model}) for persona: {persona}")
    user_weights, bounds_dict = get_user_weights_and_bounds(
        llm_client=llm_client,
        persona=persona,
        feature_mapping=feature_mapping,
        usage_tracker=usage_tracker,
    )

    elicitator = get_elicitator(
        data_path=data_path,
        user_feature_weights=user_weights,
        dataset_preset=dataset_preset,
        scaler_type=scaler_type,
    )

    step_results = []
    
    click.echo(f"Starting {num_loops} active learning loops...")
    for i in tqdm(range(num_loops), desc="LLM Elicitation"):
        candidate_pair = elicitator.select_next_candidate_pair()
        
        preference = get_llm_preference(
            llm_client=llm_client,
            persona=persona,
            candidate_pair=candidate_pair,
            usage_tracker=usage_tracker,
            feature_mapping=feature_mapping,
        )
        
        elicitator.handle_user_response(candidate_pair, preference)
        
        metrics = calculate_ranking_test_set_metrics(
            elicitator, test_items, test_utilities
        )
        metrics["loop"] = i + 1
        step_results.append(metrics)

    click.echo("\n--- Evaluation Complete ---")
    final_metrics = step_results[-1] if step_results else {}
    if final_metrics:
        click.echo(
            "Final ranking metrics: "
            + ", ".join(
                [
                    f"pairwise_acc: {final_metrics.get('pairwise_accuracy', 0):.4f}",
                    f"rho: {final_metrics.get('spearman_rho', 0):.4f}",
                    f"ndcg@1: {final_metrics.get('ndcg_1', 0):.4f}",
                    f"ndcg@3: {final_metrics.get('ndcg_3', 0):.4f}",
                    f"ndcg@full: {final_metrics.get('ndcg_full', 0):.4f}",
                ]
            )
        )

    click.echo(f"LLM Usage: {usage_tracker.request_count} requests, {usage_tracker.token_count} tokens")


@cli.command("stat")
@click.option(
    "--data-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to CSV dataset.",
)
@click.option(
    "--dataset-preset",
    type=click.Choice(["idealista-madrid", "idealista", "munich"], case_sensitive=False),
    default="idealista-madrid",
    show_default=True,
    help="Feature schema preset to apply.",
)
@click.option(
    "--scaler-type",
    type=click.Choice(["none", "minmax", "standard", "robust"], case_sensitive=False),
    default="robust",
    show_default=True,
    help="Feature scaler used during loading.",
)
@click.option(
    "--profile",
    type=str,
    default="budget_conscious",
    help="Profile providing ground truth weights.",
)
@click.option(
    "--num-loops",
    type=int,
    default=50,
    show_default=True,
    help="Number of active learning loops.",
)
@click.option(
    "--num-test-items",
    type=int,
    default=100,
    show_default=True,
    help="Size of ranking test set.",
)
@click.option(
    "--num-test-pairs",
    type=int,
    default=15,
    show_default=True,
    help="Legacy: number of pairs in pairwise test set.",
)
@click.option(
    "--runs",
    type=int,
    default=1,
    show_default=True,
    help="Number of repeated runs.",
)
@click.option(
    "--random-user-weights",
    is_flag=True,
    default=False,
    help="Use random user weights for evaluation instead of profile weights.",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    show_default=True,
    help="Global random seed.",
)
def stat(
    data_path: Path,
    dataset_preset: str,
    scaler_type: str,
    profile: str,
    num_loops: int,
    num_test_items: int,
    num_test_pairs: int,
    runs: int,
    random_user_weights: bool,
    seed: int,
) -> None:
    """Run a full statistical evaluation loop simulating a user profile."""
    click.echo(
        f"Starting {runs} run(s) - profile: {profile}, loops: {num_loops}"
    )

    for run_num in range(1, runs + 1):
        if runs > 1:
            click.echo(f"\n=== RUN {run_num}/{runs} ===")
        current_seed = seed + run_num - 1
        np.random.seed(current_seed)
        torch.manual_seed(current_seed)
        click.echo(f"Using seed: {current_seed}")

        loader = build_loader(
            data_path=data_path,
            dataset_preset=dataset_preset,
            scaler_type=scaler_type,
        )
        feature_mapping = loader.feature_mapping

        if random_user_weights:
            click.echo("Using random user weights for evaluation.")
            gt_weights = torch.tensor(
                np.random.normal(0, 0.3, size=(len(feature_mapping),)),
                dtype=torch.float32,
            )
            click.echo(f"Random user weights: {gt_weights.tolist()}")
        else:
            gt_weights, _ = load_profile_weights(profile, dataset_preset, feature_mapping)
        
        click.echo("Generating ranking test set based on ground-truth profile weights...")
        
        test_items, test_utilities = create_ranking_test_set_with_profile(
            loader.data,
            loader.scaler.inverse_transform if hasattr(loader, "scaler") and loader.scaler else lambda x: x,
            num_test_items,
            gt_weights,
            feature_mapping,
        )

        # Initialize the elicitator. We use the ground truth weights directly for the simulation.
        elicitator = get_elicitator(
            data_path=data_path,
            user_feature_weights=gt_weights,
            dataset_preset=dataset_preset,
            scaler_type=scaler_type,
        )

        step_results = []
        
        click.echo(f"Starting {num_loops} active learning loops...")
        for i in tqdm(range(num_loops), desc=f"Statistical Eval (Run {run_num})"):
            candidate_pair = elicitator.select_next_candidate_pair()
            
            # Simulate user choice using ground truth weights
            preference = get_profile_preference(candidate_pair, gt_weights)
            
            elicitator.handle_user_response(candidate_pair, preference)
            
            metrics = calculate_ranking_test_set_metrics(
                elicitator, test_items, test_utilities
            )
            metrics["loop"] = i + 1
            step_results.append(metrics)
        
        final_metrics = step_results[-1] if step_results else {}
        if final_metrics:
            click.echo(
                "Final ranking metrics: "
                + ", ".join(
                    [
                        f"pairwise_acc: {final_metrics.get('pairwise_accuracy', 0):.4f}",
                        f"rho: {final_metrics.get('spearman_rho', 0):.4f}",
                        f"ndcg@1: {final_metrics.get('ndcg_1', 0):.4f}",
                        f"ndcg@3: {final_metrics.get('ndcg_3', 0):.4f}",
                        f"ndcg@full: {final_metrics.get('ndcg_full', 0):.4f}",
                    ]
                )
            )

    if runs > 1:
        click.echo(f"\n=== ALL {runs} RUNS COMPLETED ===")


if __name__ == "__main__":
    cli()
