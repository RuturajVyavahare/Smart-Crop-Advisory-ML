"""
Crop Mapping and Normalization Module.
Provides robust canonical mapping between:
- Crop Recommendation Dataset (22 labels)
- Crop Yield Dataset (21 crops)
- India Crop Prices Dataset (122 crops)
- Cost of Cultivation Dataset (27 crops)
All crop names and descriptions are in English.
"""

from typing import Dict, List, Optional, Any
import re

# Comprehensive canonical crop dictionary in English
CROP_METADATA: Dict[str, Dict[str, Any]] = {
    "rice": {
        "canonical_name": "rice",
        "display_name": "Rice",
        "category": "Cereal",
        "yield_crop": "Rice",
        "price_crop": "Rice (Paddy)",
        "cost_crop": "Paddy",
        "default_season": "Kharif",
        "default_soil": "Inceptisols",
        "default_fertilizer": 120.0,
        "default_pesticide": 0.40,
        "description": "Major staple cereal crop thriving in warm, humid conditions with high water availability."
    },
    "maize": {
        "canonical_name": "maize",
        "display_name": "Maize",
        "category": "Cereal",
        "yield_crop": "Maize",
        "price_crop": "Maize (Corn)",
        "cost_crop": "Maize",
        "default_season": "Kharif",
        "default_soil": "Vertisols",
        "default_fertilizer": 90.0,
        "default_pesticide": 0.35,
        "description": "Versatile cereal crop requiring well-drained fertile loam and moderate rainfall."
    },
    "chickpea": {
        "canonical_name": "chickpea",
        "display_name": "Chickpea",
        "category": "Pulse",
        "yield_crop": "Chickpea",
        "price_crop": "Gram (Chana)",
        "cost_crop": "Gram",
        "default_season": "Rabi",
        "default_soil": "Vertisols",
        "default_fertilizer": 30.0,
        "default_pesticide": 0.30,
        "description": "Crucial cool-season pulse fixing atmospheric nitrogen, well suited for dryland agriculture."
    },
    "kidneybeans": {
        "canonical_name": "kidneybeans",
        "display_name": "Kidney Beans",
        "category": "Pulse",
        "yield_crop": "Minor Pulses",
        "price_crop": "Rajma (Kidney Bean)",
        "cost_crop": "Gram",
        "default_season": "Rabi",
        "default_soil": "Inceptisols",
        "default_fertilizer": 40.0,
        "default_pesticide": 0.30,
        "description": "Nutrient-rich pulse requiring mild climates and light, well-aerated soils."
    },
    "pigeonpeas": {
        "canonical_name": "pigeonpeas",
        "display_name": "Pigeon Peas (Arhar)",
        "category": "Pulse",
        "yield_crop": "Pigeonpea",
        "price_crop": "Arhar (Tur/Pigeon Pea)",
        "cost_crop": "Tur (Arhar)",
        "default_season": "Kharif",
        "default_soil": "Vertisols",
        "default_fertilizer": 35.0,
        "default_pesticide": 0.35,
        "description": "Drought-resilient deep-rooted legume widely grown across rainfed tracts."
    },
    "mothbeans": {
        "canonical_name": "mothbeans",
        "display_name": "Moth Beans",
        "category": "Pulse",
        "yield_crop": "Minor Pulses",
        "price_crop": "Moong (Green Gram)",
        "cost_crop": "Moong",
        "default_season": "Kharif",
        "default_soil": "Aridisols",
        "default_fertilizer": 20.0,
        "default_pesticide": 0.20,
        "description": "Extremely drought-hardy legume suitable for arid and semi-arid sandy regions."
    },
    "mungbean": {
        "canonical_name": "mungbean",
        "display_name": "Mung Bean (Green Gram)",
        "category": "Pulse",
        "yield_crop": "Minor Pulses",
        "price_crop": "Moong (Green Gram)",
        "cost_crop": "Moong",
        "default_season": "Kharif",
        "default_soil": "Alfisols",
        "default_fertilizer": 25.0,
        "default_pesticide": 0.25,
        "description": "Short-duration pulse ideal for crop rotation and intercropping."
    },
    "blackgram": {
        "canonical_name": "blackgram",
        "display_name": "Black Gram (Urad)",
        "category": "Pulse",
        "yield_crop": "Minor Pulses",
        "price_crop": "Urad (Black Gram)",
        "cost_crop": "Urad",
        "default_season": "Kharif",
        "default_soil": "Vertisols",
        "default_fertilizer": 25.0,
        "default_pesticide": 0.25,
        "description": "High-protein pulse adapted to warm tropical conditions with moderate moisture."
    },
    "lentil": {
        "canonical_name": "lentil",
        "display_name": "Lentil (Masur)",
        "category": "Pulse",
        "yield_crop": "Minor Pulses",
        "price_crop": "Masur (Lentil)",
        "cost_crop": "Lentil",
        "default_season": "Rabi",
        "default_soil": "Inceptisols",
        "default_fertilizer": 30.0,
        "default_pesticide": 0.25,
        "description": "Cold-tolerant winter legume requiring well-drained loamy soils."
    },
    "pomegranate": {
        "canonical_name": "pomegranate",
        "display_name": "Pomegranate",
        "category": "Horticulture",
        "yield_crop": None,
        "price_crop": "Pomegranate",
        "cost_crop": None,
        "default_season": "Whole Year",
        "default_soil": "Alfisols",
        "default_fertilizer": 150.0,
        "default_pesticide": 0.44,
        "horticultural_yield_kg_ha": 12000.0,
        "horticultural_cost_rs_ha": 95000.0,
        "description": "Commercial high-value fruit crop suitable for semi-arid subtropical zones."
    },
    "banana": {
        "canonical_name": "banana",
        "display_name": "Banana",
        "category": "Horticulture",
        "yield_crop": None,
        "price_crop": "Banana",
        "cost_crop": None,
        "default_season": "Whole Year",
        "default_soil": "Inceptisols",
        "default_fertilizer": 200.0,
        "default_pesticide": 0.44,
        "horticultural_yield_kg_ha": 35000.0,
        "horticultural_cost_rs_ha": 110000.0,
        "description": "Heavy feeder fruit crop requiring rich soil moisture and high organic content."
    },
    "mango": {
        "canonical_name": "mango",
        "display_name": "Mango",
        "category": "Horticulture",
        "yield_crop": None,
        "price_crop": "Mango",
        "cost_crop": None,
        "default_season": "Whole Year",
        "default_soil": "Alfisols",
        "default_fertilizer": 120.0,
        "default_pesticide": 0.40,
        "horticultural_yield_kg_ha": 8500.0,
        "horticultural_cost_rs_ha": 65000.0,
        "description": "Leading tropical fruit crop requiring deep well-drained alluvial soil and hot summers."
    },
    "grapes": {
        "canonical_name": "grapes",
        "display_name": "Grapes",
        "category": "Horticulture",
        "yield_crop": None,
        "price_crop": "Grapes",
        "cost_crop": None,
        "default_season": "Whole Year",
        "default_soil": "Vertisols",
        "default_fertilizer": 180.0,
        "default_pesticide": 0.44,
        "horticultural_yield_kg_ha": 20000.0,
        "horticultural_cost_rs_ha": 125000.0,
        "description": "High-capital commercial fruit requiring trellis management and warm dry summers."
    },
    "watermelon": {
        "canonical_name": "watermelon",
        "display_name": "Watermelon",
        "category": "Cucurbit",
        "yield_crop": None,
        "price_crop": "Watermelon",
        "cost_crop": None,
        "default_season": "Whole Year",
        "default_soil": "Entisols",
        "default_fertilizer": 80.0,
        "default_pesticide": 0.30,
        "horticultural_yield_kg_ha": 25000.0,
        "horticultural_cost_rs_ha": 45000.0,
        "description": "Warm-season fruit vine thriving in sandy loam riverbeds and hot days."
    },
    "muskmelon": {
        "canonical_name": "muskmelon",
        "display_name": "Muskmelon",
        "category": "Cucurbit",
        "yield_crop": None,
        "price_crop": "Muskmelon",
        "cost_crop": None,
        "default_season": "Whole Year",
        "default_soil": "Entisols",
        "default_fertilizer": 75.0,
        "default_pesticide": 0.30,
        "horticultural_yield_kg_ha": 18000.0,
        "horticultural_cost_rs_ha": 48000.0,
        "description": "Sweet fragrant summer crop demanding lots of sunshine and moderate water."
    },
    "apple": {
        "canonical_name": "apple",
        "display_name": "Apple",
        "category": "Temperate Fruit",
        "yield_crop": None,
        "price_crop": "Apple",
        "cost_crop": None,
        "default_season": "Whole Year",
        "default_soil": "Mollisols",
        "default_fertilizer": 110.0,
        "default_pesticide": 0.44,
        "horticultural_yield_kg_ha": 9500.0,
        "horticultural_cost_rs_ha": 85000.0,
        "description": "Temperate tree fruit needing chilling hours and mountain loamy soils."
    },
    "orange": {
        "canonical_name": "orange",
        "display_name": "Orange",
        "category": "Citrus",
        "yield_crop": None,
        "price_crop": "Orange",
        "cost_crop": None,
        "default_season": "Whole Year",
        "default_soil": "Vertisols",
        "default_fertilizer": 130.0,
        "default_pesticide": 0.38,
        "horticultural_yield_kg_ha": 14000.0,
        "horticultural_cost_rs_ha": 70000.0,
        "description": "Subtropical citrus crop requiring well-drained subsoils and distinct dry spells."
    },
    "papaya": {
        "canonical_name": "papaya",
        "display_name": "Papaya",
        "category": "Horticulture",
        "yield_crop": None,
        "price_crop": "Papaya",
        "cost_crop": None,
        "default_season": "Whole Year",
        "default_soil": "Inceptisols",
        "default_fertilizer": 160.0,
        "default_pesticide": 0.35,
        "horticultural_yield_kg_ha": 40000.0,
        "horticultural_cost_rs_ha": 80000.0,
        "description": "Fast-growing tropical herbaceous fruit with high fruit yield potential."
    },
    "coconut": {
        "canonical_name": "coconut",
        "display_name": "Coconut",
        "category": "Plantation",
        "yield_crop": None,
        "price_crop": "Coconut",
        "cost_crop": None,
        "default_season": "Whole Year",
        "default_soil": "Alfisols",
        "default_fertilizer": 140.0,
        "default_pesticide": 0.20,
        "horticultural_yield_kg_ha": 10500.0,
        "horticultural_cost_rs_ha": 60000.0,
        "description": "Perennial coastal palm demanding humid maritime conditions and abundant rainfall."
    },
    "cotton": {
        "canonical_name": "cotton",
        "display_name": "Cotton",
        "category": "Fiber / Cash Crop",
        "yield_crop": "Cotton",
        "price_crop": "Cotton",
        "cost_crop": "Cotton",
        "default_season": "Kharif",
        "default_soil": "Vertisols",
        "default_fertilizer": 110.0,
        "default_pesticide": 0.44,
        "description": "Cash crop excelling on black cotton soils (Vertisols) with frost-free days."
    },
    "jute": {
        "canonical_name": "jute",
        "display_name": "Jute",
        "category": "Fiber / Cash Crop",
        "yield_crop": None,
        "price_crop": "Jute",
        "cost_crop": "Jute",
        "default_season": "Kharif",
        "default_soil": "Inceptisols",
        "default_fertilizer": 60.0,
        "default_pesticide": 0.30,
        "horticultural_yield_kg_ha": 2400.0,
        "horticultural_cost_rs_ha": 42424.0,
        "description": "Golden fiber crop needing hot and humid monsoon conditions and alluvial delta soils."
    },
    "coffee": {
        "canonical_name": "coffee",
        "display_name": "Coffee",
        "category": "Plantation",
        "yield_crop": None,
        "price_crop": "Coffee",
        "cost_crop": None,
        "default_season": "Whole Year",
        "default_soil": "Alfisols",
        "default_fertilizer": 130.0,
        "default_pesticide": 0.35,
        "horticultural_yield_kg_ha": 1200.0,
        "horticultural_cost_rs_ha": 80000.0,
        "description": "Highland shade-grown plantation crop requiring rich acidic forest humus."
    }
}

# Alias dictionary for flexible normalization
CROP_ALIASES: Dict[str, str] = {
    "rice": "rice",
    "paddy": "rice",
    "dhan": "rice",
    "chawal": "rice",
    "maize": "maize",
    "corn": "maize",
    "makka": "maize",
    "chickpea": "chickpea",
    "chana": "chickpea",
    "gram": "chickpea",
    "bengal gram": "chickpea",
    "kidneybeans": "kidneybeans",
    "kidney bean": "kidneybeans",
    "rajma": "kidneybeans",
    "pigeonpeas": "pigeonpeas",
    "pigeonpea": "pigeonpeas",
    "pigeon pea": "pigeonpeas",
    "arhar": "pigeonpeas",
    "tur": "pigeonpeas",
    "toor": "pigeonpeas",
    "mothbeans": "mothbeans",
    "moth bean": "mothbeans",
    "moth dal": "mothbeans",
    "mungbean": "mungbean",
    "mung bean": "mungbean",
    "moong": "mungbean",
    "green gram": "mungbean",
    "blackgram": "blackgram",
    "black gram": "blackgram",
    "urad": "blackgram",
    "mash": "blackgram",
    "lentil": "lentil",
    "masur": "lentil",
    "masoor": "lentil",
    "pomegranate": "pomegranate",
    "anar": "pomegranate",
    "banana": "banana",
    "kela": "banana",
    "mango": "mango",
    "aam": "mango",
    "grapes": "grapes",
    "grape": "grapes",
    "angur": "grapes",
    "angoor": "grapes",
    "watermelon": "watermelon",
    "tarbooj": "watermelon",
    "tarbuz": "watermelon",
    "muskmelon": "muskmelon",
    "kharbooza": "muskmelon",
    "kharbuja": "muskmelon",
    "apple": "apple",
    "seb": "apple",
    "orange": "orange",
    "santra": "orange",
    "papaya": "papaya",
    "papita": "papaya",
    "coconut": "coconut",
    "nariyal": "coconut",
    "cotton": "cotton",
    "kapas": "cotton",
    "jute": "jute",
    "pat": "jute",
    "patson": "jute",
    "coffee": "coffee",
    "wheat": "wheat",
    "gehu": "wheat",
    "barley": "barley",
    "jau": "barley",
    "sugarcane": "sugarcane",
    "ganna": "sugarcane",
    "soybean": "soybean",
    "soyabean": "soybean",
    "groundnut": "groundnut",
    "moongfali": "groundnut",
    "sunflower": "sunflower",
    "surajmukhi": "sunflower",
    "safflower": "safflower",
    "kusum": "safflower",
    "sesame": "sesamum",
    "sesamum": "sesamum",
    "til": "sesamum",
    "bajra": "bajra",
    "pearl millet": "bajra",
    "jowar": "jowar",
    "sorghum": "jowar",
    "ragi": "ragi",
    "finger millet": "ragi"
}

def normalize_crop_name(name: str) -> str:
    """
    Normalizes any crop string into its canonical key.
    Handles capitalization, whitespace, hyphens, and aliases.
    """
    if not isinstance(name, str):
        return "rice"
    
    clean = name.strip().lower()
    clean = re.sub(r'[\(\)\[\]_/\-]', ' ', clean)
    clean = re.sub(r'\s+', ' ', clean).strip()

    # Direct check
    if clean in CROP_ALIASES:
        return CROP_ALIASES[clean]

    # Partial substring match
    for alias, canon in CROP_ALIASES.items():
        if alias in clean or clean in alias:
            return canon

    return clean

def get_crop_metadata(crop_key: str) -> Dict[str, Any]:
    """Retrieves full canonical metadata for a crop in English."""
    canon = normalize_crop_name(crop_key)
    if canon in CROP_METADATA:
        return CROP_METADATA[canon]
    
    # Generic fallback
    return {
        "canonical_name": canon,
        "display_name": canon.title(),
        "category": "Field Crop",
        "yield_crop": canon.title(),
        "price_crop": canon.title(),
        "cost_crop": canon.title(),
        "default_season": "Kharif",
        "default_soil": "Inceptisols",
        "default_fertilizer": 70.0,
        "default_pesticide": 0.35,
        "description": f"Standard agricultural crop ({canon})."
    }

def get_all_supported_crops() -> List[Dict[str, Any]]:
    """Returns a list of all 22 supported crop metadata records."""
    return list(CROP_METADATA.values())

def map_rec_crop_to_yield_crop(rec_crop: str) -> Optional[str]:
    """Maps recommendation model label to the crop name in crop_yield_dataset.csv."""
    meta = get_crop_metadata(rec_crop)
    return meta.get("yield_crop")

def map_to_price_crop(crop_name: str) -> str:
    """Maps crop name to the crop column in india-crop-prices.csv."""
    meta = get_crop_metadata(crop_name)
    return meta.get("price_crop", crop_name.title())

def map_to_cost_crop(crop_name: str) -> Optional[str]:
    """Maps crop name to crop_name column in cost-of-cultivation.csv."""
    meta = get_crop_metadata(crop_name)
    return meta.get("cost_crop")
