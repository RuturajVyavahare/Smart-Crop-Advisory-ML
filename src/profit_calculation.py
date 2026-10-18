"""
Profit Calculation and Financial Advisory Module.
Handles unit conversions (hectares, acres, quintals, kg),
revenue generation, total cultivation costs, net profit, ROI,
break-even analysis, and Indian Rupee (INR) formatting.
"""

from typing import Dict, Any, Union

HA_TO_ACRES = 2.47105
ACRES_TO_HA = 0.404686
KG_PER_QUINTAL = 100.0

def convert_area_to_ha(area_val: float, unit: str = "acre") -> float:
    """Converts input area to Hectares."""
    u = unit.strip().lower()
    if "ha" in u or "hectare" in u:
        return float(area_val)
    elif "acre" in u:
        return float(area_val) * ACRES_TO_HA
    elif "bigha" in u:
        # Standard central India bigha ~ 0.25 ha (0.62 acres)
        return float(area_val) * 0.25
    return float(area_val)

def convert_area_to_acres(area_val: float, unit: str = "acre") -> float:
    """Converts input area to Acres."""
    u = unit.strip().lower()
    if "acre" in u:
        return float(area_val)
    elif "ha" in u or "hectare" in u:
        return float(area_val) * HA_TO_ACRES
    return float(area_val)

def format_inr(number: Union[int, float], include_paise: bool = False) -> str:
    """
    Formats a number into standard Indian Rupee currency format (e.g. ₹1,25,400 or ₹1,25,400.00).
    Uses traditional lakh / crore comma separators.
    """
    if number is None:
        return "₹0"
    
    is_neg = number < 0
    num_abs = abs(number)
    
    # Split into integer and decimal parts
    parts = f"{num_abs:.2f}".split(".")
    int_part = parts[0]
    dec_part = parts[1]
    
    if len(int_part) <= 3:
        formatted = int_part
    else:
        # Last 3 digits
        last_three = int_part[-3:]
        remaining = int_part[:-3]
        
        # Group remaining digits in twos from right to left
        groups = []
        while len(remaining) > 2:
            groups.insert(0, remaining[-2:])
            remaining = remaining[:-2]
        if remaining:
            groups.insert(0, remaining)
            
        formatted = ",".join(groups) + "," + last_three

    suffix = f".{dec_part}" if include_paise else ""
    prefix = "-₹" if is_neg else "₹"
    return f"{prefix}{formatted}{suffix}"

def format_inr_short(number: Union[int, float]) -> str:
    """Short format with K, Lakh, or Crore suffix (e.g. ₹4.2 Lakh)."""
    if number is None:
        return "₹0"
    is_neg = number < 0
    val = abs(number)
    
    if val >= 10000000:
        s = f"₹{val / 10000000:.2f} Cr"
    elif val >= 100000:
        s = f"₹{val / 100000:.2f} Lakh"
    elif val >= 1000:
        s = f"₹{val / 1000:.1f} K"
    else:
        s = f"₹{val:.0f}"
        
    return f"-{s}" if is_neg else s

def calculate_farm_financials(
    yield_kg_per_ha: float,
    farm_area: float,
    area_unit: str = "acre",
    price_per_quintal: float = 3000.0,
    cost_per_ha: float = 35000.0
) -> Dict[str, Any]:
    """
    Calculates comprehensive farm economics:
    - Yield in kg and quintals per ha and acre
    - Total harvest production
    - Gross revenue
    - Total cultivation cost
    - Net profit and profit per acre/ha
    - Return on investment (ROI %)
    - Break-even yield and price thresholds
    """
    # Standardize area
    area_ha = convert_area_to_ha(farm_area, area_unit)
    area_acres = convert_area_to_acres(farm_area, area_unit)

    # Yield calculations
    yield_q_per_ha = yield_kg_per_ha / KG_PER_QUINTAL
    yield_kg_per_acre = yield_kg_per_ha / HA_TO_ACRES
    yield_q_per_acre = yield_kg_per_acre / KG_PER_QUINTAL

    # Total farm harvest
    total_yield_kg = round(yield_kg_per_ha * area_ha, 2)
    total_yield_quintals = round(total_yield_kg / KG_PER_QUINTAL, 2)

    # Price and Cost
    price_per_kg = price_per_quintal / KG_PER_QUINTAL
    cost_per_acre = cost_per_ha / HA_TO_ACRES

    # Totals
    total_revenue = round(total_yield_quintals * price_per_quintal, 2)
    total_cost = round(area_ha * cost_per_ha, 2)
    net_profit = round(total_revenue - total_cost, 2)

    # Profit metrics
    profit_per_acre = round(net_profit / area_acres, 2) if area_acres > 0 else 0.0
    profit_per_ha = round(net_profit / area_ha, 2) if area_ha > 0 else 0.0
    roi_percent = round((net_profit / total_cost) * 100.0, 2) if total_cost > 0 else 0.0
    profit_margin_percent = round((net_profit / total_revenue) * 100.0, 2) if total_revenue > 0 else 0.0

    # Break-even analysis
    be_yield_kg_per_ha = round(cost_per_ha / price_per_kg, 2) if price_per_kg > 0 else 0.0
    be_price_per_quintal = round((total_cost / total_yield_quintals), 2) if total_yield_quintals > 0 else 0.0

    return {
        # Area details
        "input_area": farm_area,
        "input_unit": area_unit,
        "area_hectares": round(area_ha, 3),
        "area_acres": round(area_acres, 3),

        # Yield breakdown
        "yield_kg_per_ha": round(yield_kg_per_ha, 2),
        "yield_quintals_per_ha": round(yield_q_per_ha, 2),
        "yield_kg_per_acre": round(yield_kg_per_acre, 2),
        "yield_quintals_per_acre": round(yield_q_per_acre, 2),
        "total_production_kg": total_yield_kg,
        "total_production_quintals": total_yield_quintals,

        # Financial rates
        "price_per_quintal": round(price_per_quintal, 2),
        "price_per_kg": round(price_per_kg, 2),
        "cost_per_ha": round(cost_per_ha, 2),
        "cost_per_acre": round(cost_per_acre, 2),

        # Financial aggregates
        "total_revenue": total_revenue,
        "total_cost": total_cost,
        "net_profit": net_profit,
        "profit_per_acre": profit_per_acre,
        "profit_per_ha": profit_per_ha,
        "roi_percent": roi_percent,
        "profit_margin_percent": profit_margin_percent,

        # Break-even indicators
        "breakeven_yield_kg_per_ha": be_yield_kg_per_ha,
        "breakeven_price_per_quintal": be_price_per_quintal,

        # Formatted currency representations
        "formatted_revenue": format_inr(total_revenue),
        "formatted_cost": format_inr(total_cost),
        "formatted_profit": format_inr(net_profit),
        "formatted_profit_per_acre": format_inr(profit_per_acre),
        "formatted_profit_per_ha": format_inr(profit_per_ha),
        "formatted_revenue_short": format_inr_short(total_revenue),
        "formatted_cost_short": format_inr_short(total_cost),
        "formatted_profit_short": format_inr_short(net_profit)
    }
