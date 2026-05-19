import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
import torch
from pydantic import BaseModel

from pairwise_bo.llm_client import LLMClientAdapter, LLMClientError, LLMUsageMetadata
from pairwise_bo.types import CandidatePair, FeatureMapping, GenericCandidate


class LowerBounds(BaseModel):
    room_number: float
    constructed_area_sqm: float


class UpperBounds(BaseModel):
    price: float
    distance_to_city_center_km: float


class FeatureWeights(BaseModel):
    price: float
    unit_price: float
    constructed_area_sqm: float
    room_number: float
    bath_number: float
    building_age_years: float
    max_building_floor: float
    dwelling_count: float
    distance_to_city_center_km: float
    distance_to_metro_km: float
    distance_to_castellana_km: float
    cadastral_quality_id: float


class LLMUserPreferences(BaseModel):
    lower_bounds: LowerBounds
    upper_bounds: UpperBounds
    feature_weights: Optional[FeatureWeights] = None
    feature_ranking: Optional[List[str]] = None


class LLMUsageTracker:
    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name
        self.request_count = 0
        self.input_count = 0
        self.output_count = 0
        self.token_count = 0

    def track_request(self, metadata: Optional[LLMUsageMetadata]) -> None:
        self.request_count += 1
        if not metadata:
            return
        if metadata.total_tokens is not None:
            self.token_count += metadata.total_tokens
        else:
            self.token_count += (metadata.prompt_tokens or 0) + (
                metadata.completion_tokens or 0
            )
        if metadata.prompt_tokens is not None:
            self.input_count += metadata.prompt_tokens
        if metadata.completion_tokens is not None:
            self.output_count += metadata.completion_tokens


def _format_candidate_for_prompt(
    candidate: GenericCandidate, name: str, feature_mapping: FeatureMapping
) -> str:
    lines = [f"Candidate {name}:"]
    candidate_dict = candidate.to_dict()
    for key, details in feature_mapping.items():
        value = candidate_dict.get(key)
        if value is None:
            continue
        if isinstance(value, float):
            if key.endswith("_score"):
                value_str = f"{(value * 100):.2f}%"
            else:
                value_str = f"{value:.2f}"
        else:
            value_str = str(value)
        unit = details.get("unit", "")
        dn = details.get("display_name", key)
        lines.append(f"- {dn}: {value_str} {unit}".strip())
    return "\n".join(lines)


def get_llm_preference(
    llm_client: LLMClientAdapter,
    persona: str,
    candidate_pair: CandidatePair,
    usage_tracker: LLMUsageTracker,
    feature_mapping: FeatureMapping,
    is_retry: bool = False,
) -> int:
    """
    Query the LLM for a preference between two candidates.
    Returns 0 if A preferred, 1 if B preferred.
    """
    prompt = f"""
Your Persona: {persona}

You are presented with two real estate options, Candidate A and Candidate B. Based on your persona, which one do you prefer?

{_format_candidate_for_prompt(candidate_pair.listing_a, "A", feature_mapping)}

{_format_candidate_for_prompt(candidate_pair.listing_b, "B", feature_mapping)}

Please state your preference by responding with only the letter 'A' or 'B'.
"""
    try:
        response = llm_client.generate_text(prompt, timeout=120)
        usage_tracker.track_request(response.usage)
        if not response or not response.text:
            raise ValueError("Empty response from LLM.")
        m = re.search(r"[AB]", response.text, re.IGNORECASE)
        if m:
            return 0 if m.group(0).upper() == "A" else 1
        click.echo("Warning: Non-A/B response. Defaulting to A.", err=True)
        return 0
    except LLMClientError as e:
        if e.retryable and not is_retry:
            click.echo("Rate limit exceeded. Waiting 60s before retry...", err=True)
            time.sleep(60)
            return get_llm_preference(
                llm_client,
                persona,
                candidate_pair,
                usage_tracker,
                feature_mapping,
                is_retry=True,
            )
        click.echo(f"Error during LLM call: {e}. Defaulting to A.", err=True)
        return 0
    except Exception as e:
        click.echo(f"LLM error: {e}. Defaulting to A.", err=True)
        return 0


def get_user_weights_and_bounds(
    llm_client: LLMClientAdapter,
    persona: str,
    feature_mapping: FeatureMapping,
    usage_tracker: LLMUsageTracker,
) -> Tuple[torch.Tensor, Dict[str, Tuple[float, float]]]:
    """
    Obtain initial user bounds and feature weights from the LLM.
    Returns weights tensor and a parsed bounds dictionary.
    """
    considered_features = list(feature_mapping.keys())

    feature_instruction_block = f"""
3. Guess the user's weight for a set of {len(considered_features)} features. The weights represent importance for apartment selection.
Weights can be positive (benefit) or negative (cost) and the sum of absolute values MUST equal 1.
These are the features to weight: {", ".join(considered_features)}.
Provide them as "feature_weights" JSON object (required!) with numeric values.
"""

    prompt = f"""You are a real estate agent. Interview a user, who is looking to buy a new real estate property. Your goal is to find out what the user values most and which criteria are important for them.

User persona:
"{persona}"

There are four main outcomes you should return:

1. Lower bounds on the following criteria:
- Size of the constructed area in square meters (constructed_area_sqm)
- Number of rooms in the property (room_number)

2. Upper bounds for the following criteria:
- Total purchasing price with everything included (price)
- Distance to the city center in km (distance_to_city_center_km)

{feature_instruction_block}

Based on the provided user profile, please return JSON that describes the collected information you are certain about.
"""
    response = llm_client.generate_json(
        prompt,
        schema=LLMUserPreferences,
        temperature=0.8,
    )
    usage_tracker.track_request(response.usage)
    if not isinstance(response.parsed, LLMUserPreferences):
        raise ValueError("LLM response missing parsed user preferences.")

    user_preferences = response.parsed

    # Build weights tensor in the exact order of feature_mapping index
    if user_preferences.feature_weights is None:
        raise ValueError("LLM response missing 'feature_weights'.")

    # Map weight values to list
    weight_dict = user_preferences.feature_weights.model_dump()
    weights_list = [0.0] * len(considered_features)
    for key, val in weight_dict.items():
        if key in feature_mapping:
            idx = feature_mapping[key]["index"]
            weights_list[idx] = float(val)

    weights_tensor = torch.tensor(weights_list, dtype=torch.float32)

    # Normalization (sum of absolute values == 1)
    abs_sum = torch.sum(torch.abs(weights_tensor))
    if abs_sum > 0:
        weights_tensor = weights_tensor / abs_sum

    # Parsed bounds
    bounds_dict = {}
    if user_preferences.lower_bounds.room_number > 0:
        bounds_dict["room_number"] = (float(user_preferences.lower_bounds.room_number), float("inf"))
    if user_preferences.lower_bounds.constructed_area_sqm > 0:
        bounds_dict["constructed_area_sqm"] = (float(user_preferences.lower_bounds.constructed_area_sqm), float("inf"))
    
    if user_preferences.upper_bounds.price > 0:
        bounds_dict["price"] = (0.0, float(user_preferences.upper_bounds.price))
    if user_preferences.upper_bounds.distance_to_city_center_km > 0:
        bounds_dict["distance_to_city_center_km"] = (0.0, float(user_preferences.upper_bounds.distance_to_city_center_km))

    return weights_tensor, bounds_dict
