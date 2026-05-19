"""Built-in feature schema presets for common datasets."""

from pairwise_bo.types import FeatureMapping


MUNICH_FEATURE_MAPPING: FeatureMapping = {
    "total_rent": {"index": 0, "display_name": "Total Rent", "unit": "EUR"},
    "floor": {"index": 1, "display_name": "Floor", "unit": ""},
    "living_area_sqm": {"index": 2, "display_name": "Living Area", "unit": "sqm"},
    "parking_spaces": {"index": 3, "display_name": "Parking Spaces", "unit": "spaces"},
    "outdoor_leisure_score": {"index": 4, "display_name": "Outdoor Leisure Score", "unit": ""},
    "recreation_dining_score": {"index": 5, "display_name": "Recreation Dining Score", "unit": ""},
    "bikeability_score": {"index": 6, "display_name": "Bikeability Score", "unit": ""},
    "noise_score": {"index": 7, "display_name": "Noise Score", "unit": ""},
    "safety_score": {"index": 8, "display_name": "Safety Score", "unit": ""},
    "travel_time_public_transport": {
        "index": 9,
        "display_name": "Travel Time Public Transport",
        "unit": "seconds",
    },
    "travel_time_grocery_store": {
        "index": 10,
        "display_name": "Travel Time Grocery Store",
        "unit": "seconds",
    },
    "travel_time_outdoor_leisure": {
        "index": 11,
        "display_name": "Travel Time Outdoor Leisure",
        "unit": "seconds",
    },
    "travel_time_city_center": {
        "index": 12,
        "display_name": "Travel Time City Center",
        "unit": "seconds",
    },
}


IDEALISTA_FEATURE_MAPPING: FeatureMapping = {
    "price": {"index": 0, "display_name": "Price", "unit": "EUR"},
    "unit_price": {"index": 1, "display_name": "Unit Price", "unit": "EUR/sqm"},
    "constructed_area_sqm": {"index": 2, "display_name": "Constructed Area", "unit": "sqm"},
    "room_number": {"index": 3, "display_name": "Rooms", "unit": "rooms"},
    "bath_number": {"index": 4, "display_name": "Bathrooms", "unit": "bathrooms"},
    "building_age_years": {"index": 5, "display_name": "Building Age", "unit": "years"},
    "max_building_floor": {"index": 6, "display_name": "Max Building Floor", "unit": "floors"},
    "dwelling_count": {"index": 7, "display_name": "Dwelling Count", "unit": "units"},
    "distance_to_city_center_km": {
        "index": 8,
        "display_name": "Distance to City Center",
        "unit": "km",
    },
    "distance_to_metro_km": {"index": 9, "display_name": "Distance to Metro", "unit": "km"},
    "distance_to_castellana_km": {
        "index": 10,
        "display_name": "Distance to Castellana",
        "unit": "km",
    },
    "cadastral_quality_id": {
        "index": 11,
        "display_name": "Cadastral Quality ID",
        "unit": "id",
    },
}


IDEALISTA_RENAME_MAP: dict[str, str] = {
    "PRICE": "price",
    "UNITPRICE": "unit_price",
    "CONSTRUCTEDAREA": "constructed_area_sqm",
    "ROOMNUMBER": "room_number",
    "BATHNUMBER": "bath_number",
    "AGE": "building_age_years",
    "CADMAXBUILDINGFLOOR": "max_building_floor",
    "CADDWELLINGCOUNT": "dwelling_count",
    "DISTANCE_TO_CITY_CENTER": "distance_to_city_center_km",
    "DISTANCE_TO_METRO": "distance_to_metro_km",
    "DISTANCE_TO_CASTELLANA": "distance_to_castellana_km",
    "CADASTRALQUALITYID": "cadastral_quality_id",
}
