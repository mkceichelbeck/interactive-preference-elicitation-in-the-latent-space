"""High-level helper functions for iterative BO workflows."""

from __future__ import annotations

from typing import Optional

import torch

from pairwise_bo.core import PreferenceElicitator
from pairwise_bo.types import CandidatePair


def run_single_feedback_step(
    elicitator: PreferenceElicitator,
    response: Optional[int] = None,
) -> CandidatePair:
    """Run one BO iteration and update state with a response or model-predicted choice."""
    candidate_pair = elicitator.select_next_candidate_pair()
    if response is None:
        response = elicitator.predict_choice(candidate_pair.to_tensor())

    elicitator.handle_user_response(candidate_pair, int(response))
    return candidate_pair


def rank_batch(
    elicitator: PreferenceElicitator,
    listings: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ranking indices and sorted utility scores for a listing batch."""
    indices, scores = elicitator.rank_listings(listings, return_scores=True)
    return indices, scores
