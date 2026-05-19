"""Core Pairwise GP Bayesian optimization implementation."""

from __future__ import annotations

import os
from itertools import combinations
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from botorch.acquisition import AnalyticExpectedUtilityOfBestOption
from botorch.fit import fit_gpytorch_mll
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.optim import optimize_acqf

from pairwise_bo.autoencoder import Autoencoder
from pairwise_bo.data import CsvDatasetLoader
from pairwise_bo.logging_utils import setup_logger
from pairwise_bo.types import CandidatePair, GenericCandidate

logger = setup_logger(__name__)

SMOKE_TEST = os.environ.get("SMOKE_TEST")
NUM_RESTARTS = 40 if not SMOKE_TEST else 4
RAW_SAMPLES = 512 if not SMOKE_TEST else 16
PAIR_QUERY_SIZE = 2


class PreferenceElicitator:
    """Pairwise preference elicitation using a Pairwise GP model."""

    def __init__(
        self,
        data_loader: CsvDatasetLoader,
        user_weights: Optional[torch.Tensor] = None,
        bounds: Optional[torch.Tensor] = None,
        saved_model_path: Optional[Path] = None,
    ):
        logger.debug("Initializing PreferenceElicitator")

        self.dl = data_loader
        self.data = self.dl.data
        self.feature_keys = self.dl.feature_keys
        self.dim = int(self.data.shape[1])

        self.user_weights: Optional[torch.Tensor] = None
        if user_weights is not None:
            if user_weights.numel() != self.dim:
                raise ValueError(
                    f"user_weights dimension mismatch. Expected {self.dim}, got {user_weights.numel()}."
                )
            self.user_weights = user_weights.to(dtype=torch.float32)

        if bounds is not None:
            if bounds.shape != (2, self.dim):
                raise ValueError(
                    f"Bounds must have shape (2, {self.dim}), got {tuple(bounds.shape)}"
                )
            bounds = bounds.float()
            if self.dl.has_scaler():
                bounds = self.dl.scale(bounds)
            self.bounds = bounds
        else:
            self.bounds = self.dl.get_bounds()

        self.bounds = self.bounds.double()

        if saved_model_path is not None:
            if not saved_model_path.exists():
                raise FileNotFoundError(
                    f"Model path does not exist: {saved_model_path}"
                )
            self._load_model(saved_model_path)
        else:
            self._init_model(n=1)

    @property
    def total_compare_count(self) -> int:
        return int(self.train_comps.shape[0])

    def _init_model(self, n: int) -> None:
        train_vals, train_comps = self.generate_random_pref_data(n)
        if not hasattr(self, "train_vals") or not hasattr(self, "train_comps"):
            self.train_vals = train_vals
            self.train_comps = train_comps
        else:
            self.train_comps = torch.cat(
                (self.train_comps, train_comps + self.train_vals.shape[0]), dim=0
            )
            self.train_vals = torch.cat((self.train_vals, train_vals), dim=0)
        self.model = fit_pref_model(self.train_vals, self.train_comps, self.bounds)

    def generate_random_pref_data(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate n synthetic pairwise comparisons from current dataset."""
        if self.user_weights is None:
            raise ValueError(
                "user_weights must be provided to generate initial preference comparisons."
            )

        n_candidates = min(self.data.shape[0], max(2, n * 2))
        rand_indices = torch.randperm(self.data.shape[0])[:n_candidates]
        x = self.data[rand_indices].float()

        x_raw = self.dl.reverse_scaling(x.numpy()) if self.dl.has_scaler() else x.numpy()
        util = self._util_func(torch.tensor(x_raw))

        max_pairs = (x.shape[0] * (x.shape[0] - 1)) // 2
        n_comps = min(n, max_pairs)
        if n_comps <= 0:
            raise ValueError("Not enough candidates to generate pairwise comparisons.")

        comps = gen_comps(util, n_comps=n_comps)
        return x, comps

    def select_next_candidate_pair(self) -> CandidatePair:
        """Choose the next pair of candidates using EUBO acquisition."""
        acq_func = AnalyticExpectedUtilityOfBestOption(pref_model=self.model)

        data_d = self.data.to(dtype=self.bounds.dtype)
        valid_mask = torch.all(
            (data_d >= self.bounds[0]) & (data_d <= self.bounds[1]),
            dim=1,
        )
        valid_data = data_d[valid_mask]

        if valid_data.shape[0] == 0:
            logger.warning("No points satisfy bounds, sampling from full set and clamping.")
            initial_conditions = data_d[
                torch.randint(0, data_d.shape[0], (NUM_RESTARTS, PAIR_QUERY_SIZE))
            ]
            initial_conditions = torch.clamp(
                initial_conditions,
                min=self.bounds[0],
                max=self.bounds[1],
            )
        else:
            initial_conditions = valid_data[
                torch.randint(0, valid_data.shape[0], (NUM_RESTARTS, PAIR_QUERY_SIZE))
            ]

        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=self.bounds,
            q=PAIR_QUERY_SIZE,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
            batch_initial_conditions=initial_conditions,
        )

        cand = candidates.detach().to(dtype=torch.float32)

        means, stds = self._predict_with_uncertainty(cand, normalize=False)
        candidate_1 = GenericCandidate.from_numpy(
            cand[0].cpu().numpy(),
            self.feature_keys,
        )
        candidate_2 = GenericCandidate.from_numpy(
            cand[1].cpu().numpy(),
            self.feature_keys,
        )

        candidate_1.preference_mean = float(means[0].item())
        candidate_1.preference_std = float(stds[0].item())
        candidate_2.preference_mean = float(means[1].item())
        candidate_2.preference_std = float(stds[1].item())

        return CandidatePair(listing_a=candidate_1, listing_b=candidate_2)

    def handle_user_response(self, candidates: CandidatePair, response: int) -> None:
        """Update model observations with one user choice over a candidate pair."""
        candidates_t = candidates.to_tensor()
        self.update_observations(candidates_t, response)

    def update_observations(self, candidates: torch.Tensor, response: int) -> None:
        if response not in (0, 1):
            raise ValueError("response must be 0 or 1.")

        comp = torch.tensor([[response, 1 - response]], dtype=torch.long)
        self.train_comps = torch.cat(
            (self.train_comps, comp + self.train_vals.shape[0]),
            dim=0,
        )
        self.train_vals = torch.cat((self.train_vals, candidates), dim=0)

        self.model = fit_pref_model(self.train_vals, self.train_comps, self.bounds)

    def save_model(self, path: Path) -> None:
        """Persist model state and training tensors to disk."""
        if path.suffix == ".pt":
            path = path.with_suffix("")

        torch.save(self.model.state_dict(), path.with_suffix(".pt"))
        torch.save(
            self.train_vals,
            path.with_stem(path.stem + "_train_vals").with_suffix(".pt"),
        )
        torch.save(
            self.train_comps,
            path.with_stem(path.stem + "_train_comps").with_suffix(".pt"),
        )

    def _load_model(self, path: Path) -> None:
        """Load persisted training tensors and rebuild posterior model."""
        if path.suffix == ".pt":
            path = path.with_suffix("")

        self.train_vals = torch.load(
            path.with_stem(path.stem + "_train_vals").with_suffix(".pt")
        )
        self.train_comps = torch.load(
            path.with_stem(path.stem + "_train_comps").with_suffix(".pt")
        )
        self.model = fit_pref_model(self.train_vals, self.train_comps, self.bounds)

    def _util_func(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        if self.user_weights is None:
            raise ValueError("user_weights must be provided for utility calculation.")
        return (x * self.user_weights).sum(dim=-1)

    def _predict_with_uncertainty(
        self,
        x: torch.Tensor,
        normalize: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        if self.model is None:
            raise ValueError("Model is not trained.")

        self.model.eval()

        with torch.no_grad():
            posterior = self.model.posterior(x.to(dtype=torch.double))
            mvn = posterior.mvn  # type: ignore[assignment]
            mean = mvn.mean.squeeze()
            std = mvn.variance.sqrt().squeeze()

        if normalize:
            mean_range = mean.max() - mean.min()
            if mean_range > 0:
                mean = (mean - mean.min()) / mean_range

            std_range = std.max() - std.min()
            if std_range > 0:
                std = (std - std.min()) / std_range

        return mean, std

    def rank_listings(
        self,
        listings: torch.Tensor,
        return_scores: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Rank listings by posterior utility mean in descending order."""
        if len(listings.shape) == 1:
            listings = listings.unsqueeze(0)

        scores, _ = self._predict_with_uncertainty(listings, normalize=False)
        sorted_indices = torch.argsort(scores, descending=True)

        if return_scores:
            return sorted_indices, scores[sorted_indices]
        return sorted_indices

    def _posterior_two(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, object]:
        self.model.eval()
        dtype = self.train_vals.dtype if self.train_vals is not None else torch.float32
        stacked = torch.stack([x_a, x_b], dim=0).to(dtype=dtype)
        with torch.no_grad():
            post = self.model.posterior(stacked)

        mu = post.mean.squeeze(-1)  # type: ignore[attr-defined]
        cov = post.mvn.covariance_matrix  # type: ignore[attr-defined]

        mu_delta = mu[0] - mu[1]
        var_delta = cov[0, 0] + cov[1, 1] - (2 * cov[0, 1])
        var_delta = torch.clamp(var_delta, min=0.0)
        return mu_delta, var_delta, post

    def preference_probability_mc(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        n_samples: int = 512,
        integrate_logistic: bool = False,
        seed: int | None = None,
    ) -> float:
        """Monte-Carlo estimate of P(a preferred to b)."""
        if seed is not None:
            torch.manual_seed(seed)

        _, _, post = self._posterior_two(x_a, x_b)
        samples = post.rsample(sample_shape=torch.Size([n_samples]))
        delta_samples = samples[:, 0] - samples[:, 1]
        if integrate_logistic:
            probability = torch.sigmoid(delta_samples).mean()
        else:
            probability = (delta_samples > 0).float().mean()
        return float(probability.item())

    def predict_choice(self, candidates: torch.Tensor) -> int:
        """Predict argmax preference between two candidate vectors."""
        probability = self.preference_probability_mc(
            x_a=candidates[0],
            x_b=candidates[1],
            n_samples=512,
            integrate_logistic=False,
        )
        return 0 if probability >= 0.5 else 1


class AutoencoderPreferenceElicitator(PreferenceElicitator):
    """Preference elicitator operating in AE latent space."""

    def __init__(
        self,
        autoencoder_model: Autoencoder,
        latent_dim: int,
        data_loader: CsvDatasetLoader,
        user_weights: Optional[torch.Tensor] = None,
        bounds: Optional[torch.Tensor] = None,
        saved_model_path: Optional[Path] = None,
    ):
        self.autoencoder = autoencoder_model
        super().__init__(
            data_loader=data_loader,
            user_weights=user_weights,
            bounds=bounds,
            saved_model_path=saved_model_path,
        )

        self.dim = latent_dim
        self.data_bounds = self.bounds

        if bounds is None or self.bounds.shape != (2, self.dim):
            encoded_data = self._encode(self.data)
            min_x, _ = encoded_data.min(dim=0)
            max_x, _ = encoded_data.max(dim=0)
            self.bounds = torch.stack([min_x, max_x], dim=0).float()

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        with torch.no_grad():
            return self.autoencoder.encode(x).detach()

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        if len(z.shape) == 1:
            z = z.unsqueeze(0)
        with torch.no_grad():
            reconstructed = self.autoencoder.decode(z)
        return reconstructed.detach().float()

    def generate_random_pref_data(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.user_weights is None:
            raise ValueError(
                "user_weights must be provided to generate initial preference comparisons."
            )

        n_candidates = min(self.data.shape[0], max(2, n * 2))
        rand_indices = torch.randperm(self.data.shape[0])[:n_candidates]
        x = self.data[rand_indices].float()
        x_raw = self.dl.reverse_scaling(x.numpy()) if self.dl.has_scaler() else x.numpy()
        util = self._util_func(torch.tensor(x_raw))

        encoded_x = self._encode(x)
        max_pairs = (encoded_x.shape[0] * (encoded_x.shape[0] - 1)) // 2
        n_comps = min(n, max_pairs)
        if n_comps <= 0:
            raise ValueError("Not enough candidates to generate pairwise comparisons.")

        comps = gen_comps(util, n_comps=n_comps)
        return encoded_x, comps

    def select_next_candidate_pair(self) -> CandidatePair:
        acq_func = AnalyticExpectedUtilityOfBestOption(pref_model=self.model)

        if self.data_bounds is None:
            raise ValueError("data_bounds is not initialized.")

        valid_mask = torch.all(
            (self.data >= self.data_bounds[0]) & (self.data <= self.data_bounds[1]),
            dim=1,
        )
        valid_data = self.data[valid_mask]

        if valid_data.shape[0] == 0:
            logger.warning("No points satisfy data bounds, using full set with clamping.")
            initial_conditions = self.data[
                torch.randint(0, self.data.shape[0], (NUM_RESTARTS, PAIR_QUERY_SIZE))
            ]
            initial_conditions = torch.clamp(
                initial_conditions,
                min=self.data_bounds[0],
                max=self.data_bounds[1],
            )
        else:
            initial_conditions = valid_data[
                torch.randint(0, valid_data.shape[0], (NUM_RESTARTS, PAIR_QUERY_SIZE))
            ]

        encoded_initial_conditions = self._encode(initial_conditions)
        encoded_initial_conditions = torch.clamp(
            encoded_initial_conditions,
            min=self.bounds[0],
            max=self.bounds[1],
        ).to(dtype=self.bounds.dtype)

        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=self.bounds.to(dtype=torch.double),
            q=PAIR_QUERY_SIZE,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
            batch_initial_conditions=encoded_initial_conditions,
        )

        means, stds = self._predict_with_uncertainty(candidates, normalize=False)

        decoded_candidates = self._decode(candidates.float())
        candidate_1 = GenericCandidate.from_numpy(
            decoded_candidates[0].cpu().numpy(),
            self.feature_keys,
        )
        candidate_2 = GenericCandidate.from_numpy(
            decoded_candidates[1].cpu().numpy(),
            self.feature_keys,
        )

        candidate_1.preference_mean = float(means[0].item())
        candidate_1.preference_std = float(stds[0].item())
        candidate_2.preference_mean = float(means[1].item())
        candidate_2.preference_std = float(stds[1].item())

        return CandidatePair(listing_a=candidate_1, listing_b=candidate_2)

    def handle_user_response(self, candidates: CandidatePair, response: int) -> None:
        candidates_t = candidates.to_tensor()
        encoded_candidates = self._encode(candidates_t)
        self.update_observations(encoded_candidates, response)

    def rank_listings(
        self,
        listings: torch.Tensor,
        return_scores: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        encoded_listings = self._encode(listings)
        return super().rank_listings(encoded_listings, return_scores=return_scores)

    def predict_choice(self, candidates: torch.Tensor) -> int:
        encoded_candidates = self._encode(candidates)
        return super().predict_choice(encoded_candidates)


def fit_pref_model(
    x: torch.Tensor,
    comp: torch.Tensor,
    bounds: torch.Tensor,
) -> PairwiseGP:
    """Fit a Pairwise GP model to candidate vectors and pairwise outcomes."""
    del bounds

    model = PairwiseGP(
        datapoints=x.double(),
        comparisons=comp.long(),
    )
    model.train()
    mll = PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model)
    fitted_mll = fit_gpytorch_mll(mll)
    if fitted_mll.training:
        logger.warning("Pairwise GP fitting did not fully converge.")
    return model


def gen_comps(utils: torch.Tensor, n_comps: int, noise: float = 0.1) -> torch.Tensor:
    """Generate pairwise preference comparisons from utility values."""
    all_pairs = np.array(list(combinations(range(utils.shape[0]), 2)))
    if n_comps > len(all_pairs):
        raise ValueError(
            f"Requested {n_comps} comparisons but only {len(all_pairs)} unique pairs exist."
        )

    comp_pairs = all_pairs[np.random.choice(len(all_pairs), n_comps, replace=False)]
    c0 = utils[comp_pairs[:, 0]] + np.random.standard_normal(len(comp_pairs)) * noise
    c1 = utils[comp_pairs[:, 1]] + np.random.standard_normal(len(comp_pairs)) * noise

    reverse = (c0 < c1).numpy()
    comp_pairs[reverse, :] = np.flip(comp_pairs[reverse, :], 1)
    return torch.tensor(comp_pairs, dtype=torch.long)
