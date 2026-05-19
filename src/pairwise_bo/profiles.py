import torch
from typing import Tuple

PERSONAS = {
    "family": "You are the head of a family with two young children. You prioritize space, multiple rooms and bathrooms, and high-quality housing. You value properties with more floors in the building for better amenities. You can afford higher prices but want good value per square meter. Distance to city center is less important than space and quality.",
    "student": "You are a university student on a tight budget. Low price is your absolute top priority, and you're willing to accept smaller space and fewer rooms. You prefer being close to the city center and metro stations for easy access to university and nightlife. You don't mind older buildings or lower quality if it means lower costs.",
    "young_professional": "You are a young professional who values convenience and modern living. You prioritize proximity to metro stations and reasonable distance to city center for your commute. You prefer newer buildings with good quality, and you're willing to pay higher prices per square meter for better location and quality. Moderate space requirements are sufficient.",
    "noise_averse": "You prioritize peaceful living and prefer properties farther from the busy city center and metro stations to avoid noise. You value higher floors in buildings for reduced street noise, and you're willing to pay premium prices for tranquil locations. Building quality and moderate space are important, but distance from transportation hubs is preferred for quieter environment.",
}

PBO_PROFILES = {
    "budget_conscious": {
        "weights": {
            "total_rent": -0.5,
            "living_area_sqm": 0.1,
            "parking_spaces": 0.05,
            "travel_time_public_transport": -0.2,
            "travel_time_grocery_store": -0.05,
            "travel_time_city_center": -0.1,
        },
        "bounds": {},
    },
    "urban_commuter": {
        "weights": {
            "travel_time_public_transport": -0.3,
            "travel_time_city_center": -0.3,
            "bikeability_score": 0.2,
            "total_rent": -0.1,
            "recreation_dining_score": 0.05,
            "safety_score": 0.05,
        },
    },
    "noise_averse": {
        "weights": {
            "noise_score": 0.4,
            "safety_score": 0.2,
            "travel_time_outdoor_leisure": -0.2,
            "travel_time_city_center": -0.1,
            "total_rent": -0.1,
        },
    },
    "family_friendly": {
        "weights": {
            "living_area_sqm": 0.3,
            "safety_score": 0.3,
            "parking_spaces": 0.1,
            "recreation_dining_score": 0.1,
            "travel_time_grocery_store": -0.1,
            "total_rent": -0.1,
            "floor": -0.05,
            "noise_score": -0.05,
        },
    },
}

DEFAULT_PBO_USER_WEIGHTS = torch.tensor(
    [
        -0.25,  # total_rent
        0.02,   # floor
        0.08,   # living_area_sqm
        -0.01,  # parking_space
        0.07,   # outdoor_leisure_score
        0.06,   # recreation_dining_score
        0.05,   # bikeability_score
        -0.04,  # noise_score
        0.10,   # safety_score
        -0.15,  # travel_time_public_transport
        -0.06,  # travel_time_grocery_store
        -0.03,  # travel_time_outdoor_leisure
        -0.06,  # travel_time_city_center
    ]
)

PROFILES_IDEALISTA = {
    "budget_conscious": {
        "weights": {
            "price": -0.50,
            "constructed_area_sqm": 0.10,
            "room_number": 0.05,
            "distance_to_metro_km": -0.20,
            "distance_to_castellana_km": -0.05,
            "distance_to_city_center_km": -0.10,
        },
    },
    "urban_commuter": {
        "weights": {
            "distance_to_metro_km": -0.30,
            "distance_to_city_center_km": -0.30,
            "unit_price": -0.10,
            "constructed_area_sqm": 0.05,
            "dwelling_count": 0.05,
        },
    },
    "noise_averse": {
        "weights": {
            "distance_to_castellana_km": 0.40,
            "building_age_years": -0.10,
            "distance_to_metro_km": -0.20,
            "distance_to_city_center_km": 0.10,
            "price": -0.10,
        },
    },
    "family_friendly": {
        "weights": {
            "constructed_area_sqm": 0.30,
            "room_number": 0.20,
            "bath_number": 0.10,
            "price": -0.10,
            "distance_to_city_center_km": -0.10,
            "max_building_floor": -0.05,
            "distance_to_castellana_km": 0.05,
        },
    },
}

DEFAULT_IDEALISTA_USER_WEIGHTS = torch.tensor(
    [
        -0.3,   # total_rent
        0.0,    # unitprice
        0.2,    # constructed_area_sqm
        0.1,    # number_of_rooms
        0.05,   # bathnumber
        -0.1,   # building age
        0.01,   # max building floors
        -0.01,  # dwelling count
        -0.1,   # distance to city center
        -0.10,  # distance to metro
        -0.03,  # distance to castellana
        0.0,    # cadastralquality
    ]
)

def load_profile_weights(
    profile_name: str, preset: str, feature_mapping: dict
) -> Tuple[torch.Tensor, dict]:
    lowered = preset.lower()
    if lowered == "idealista-madrid" or lowered == "idealista":
        profiles = PROFILES_IDEALISTA
        default_weights = DEFAULT_IDEALISTA_USER_WEIGHTS
    elif lowered == "munich":
        profiles = PBO_PROFILES
        default_weights = DEFAULT_PBO_USER_WEIGHTS
    else:
        raise ValueError(f"Unknown preset {preset} for loading profile weights.")

    if profile_name not in profiles:
        print(f"Warning: Profile '{profile_name}' not found. Using default weights.")
        return default_weights, {}

    prof_data = profiles[profile_name]
    weight_dict = prof_data.get("weights", {})
    bounds_dict = prof_data.get("bounds", {})

    weights_list = [0.0] * len(feature_mapping)
    for key, val in weight_dict.items():
        if key in feature_mapping:
            idx = feature_mapping[key]["index"]
            weights_list[idx] = float(val)

    weights_tensor = torch.tensor(weights_list, dtype=torch.float32)
    # Normalize absolute sum to 1
    abs_sum = torch.sum(torch.abs(weights_tensor))
    if abs_sum > 0:
        weights_tensor = weights_tensor / abs_sum

    return weights_tensor, bounds_dict
