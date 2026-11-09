"""Derived oceanographic features and the natural-language summary that gets embedded."""
from __future__ import annotations

import math

import numpy as np

from .netcdf import Profile

# Coarse named regions (lat range, lon range) used to describe profiles and to parse questions.
REGIONS: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "Arabian Sea": ((5, 26), (50, 78)),
    "Bay of Bengal": ((5, 24), (78, 100)),
    "Andaman Sea": ((5, 20), (92, 100)),
    "Mozambique Channel": ((-27, -10), (32, 46)),
    "Eastern Indian Ocean off Australia": ((-36, -10), (100, 120)),
    "Equatorial Indian Ocean": ((-5, 5), (40, 110)),
    "Southern Indian Ocean": ((-60, -30), (20, 120)),
}


def region_of(lat: float, lon: float) -> str:
    for name, ((a, b), (c, d)) in REGIONS.items():
        if a <= lat <= b and c <= lon <= d:
            return name
    return "Indian Ocean (open water)" if lat < 0 else "northern Indian Ocean"


def _near_surface(pres: np.ndarray, val: np.ndarray, max_dbar: float = 25.0) -> float | None:
    idx = np.where(~np.isnan(val) & (pres <= max_dbar))[0]
    if idx.size == 0:
        return None
    return float(val[idx[np.argmin(pres[idx])]])


def mixed_layer_depth(pres: np.ndarray, temp: np.ndarray, delta: float = 0.2, ref_dbar: float = 10.0) -> float | None:
    """Depth where temperature first drops `delta` degC below its value at `ref_dbar` (de Boyer Montegut style)."""
    ok = ~np.isnan(temp) & ~np.isnan(pres)
    p, t = pres[ok], temp[ok]
    if p.size < 3 or p.min() > ref_dbar + 5:
        return None
    t_ref = float(np.interp(ref_dbar, p, t))
    below = np.where((p > ref_dbar) & (t <= t_ref - delta))[0]
    if below.size == 0:
        return None
    j = below[0]
    # linear interpolation between the bracketing levels
    p0, t0 = (p[j - 1], t[j - 1]) if j > 0 else (ref_dbar, t_ref)
    return float(p0 + (t_ref - delta - t0) * (p[j] - p0) / (t[j] - t0)) if t[j] != t0 else float(p[j])


def thermocline_depth(pres: np.ndarray, temp: np.ndarray, min_dz: float = 5.0) -> float | None:
    """Mid-depth of the layer with the largest |dT/dz| (levels closer than `min_dz` dbar are skipped)."""
    ok = ~np.isnan(temp) & ~np.isnan(pres)
    p, t = pres[ok], temp[ok]
    if p.size < 3:
        return None
    dz = np.diff(p)
    valid = dz >= min_dz
    if not valid.any():
        return None
    grad = np.abs(np.diff(t)) / np.where(valid, dz, np.nan)
    j = int(np.nanargmax(grad))
    return float((p[j] + p[j + 1]) / 2)


def features(p: Profile) -> dict:
    return {
        "sst": _near_surface(p.pres, p.temp),
        "sss": _near_surface(p.pres, p.psal),
        "mld_dbar": mixed_layer_depth(p.pres, p.temp),
        "thermocline_dbar": thermocline_depth(p.pres, p.temp),
        "max_pres_dbar": float(np.nanmax(p.pres)),
    }


def _band(v: float | None, cuts: tuple[float, ...], words: tuple[str, ...]) -> str:
    if v is None:
        return ""
    for c, w in zip(cuts, words):
        if v < c:
            return w
    return words[-1]


def summarize(p: Profile, f: dict) -> str:
    """Describe a profile in plain language. This text is what gets embedded, so it deliberately uses the
    words a person would ask with (warm, fresh, saline, shallow mixed layer, region names, month)."""
    parts = [
        f"Argo float {p.wmo_id} cycle {p.cycle} on {p.time:%d %B %Y} at {abs(p.lat):.1f}°{'N' if p.lat >= 0 else 'S'} "
        f"{abs(p.lon):.1f}°E in the {region_of(p.lat, p.lon)}."
    ]
    if f["sst"] is not None:
        parts.append(f"Surface temperature {f['sst']:.1f} °C, "
                     f"{_band(f['sst'], (15, 22, 27), ('cold', 'cool', 'warm', 'very warm'))} water.")
    if f["sss"] is not None:
        river = " river-runoff influenced" if f["sss"] < 33.5 and region_of(p.lat, p.lon) in ("Bay of Bengal", "Andaman Sea") else ""
        parts.append(f"Surface salinity {f['sss']:.2f} PSU: "
                     f"{_band(f['sss'], (33.5, 35, 36), ('fresh' + river, 'moderately fresh', 'ordinary', 'salty, evaporation-dominated'))} water.")
    if f["mld_dbar"] is not None:
        parts.append(f"Mixed layer about {f['mld_dbar']:.0f} dbar deep "
                     f"({_band(f['mld_dbar'], (25, 60), ('shallow', 'moderate', 'deep'))}).")
    if f["thermocline_dbar"] is not None:
        parts.append(f"Sharpest thermocline near {f['thermocline_dbar']:.0f} dbar.")
    parts.append(f"Profile reaches {f['max_pres_dbar']:.0f} dbar.")
    return " ".join(parts)


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = math.pi / 180
    a = math.sin((lat2 - lat1) * r / 2) ** 2 + math.cos(lat1 * r) * math.cos(lat2 * r) * math.sin((lon2 - lon1) * r / 2) ** 2
    return 12742 * math.asin(math.sqrt(a))


# Words in a question that translate to a numeric constraint on a stored feature. Used for hybrid retrieval:
# the semantic ranker works on meaning, but embeddings blur antonyms ("low-salinity" ~ "high-salinity"),
# so the constraint is applied exactly and the embedding ranks what remains.
FACETS: list[tuple[str, str, str, float]] = [  # (regex, column, op, value)
    (r"low[- ]salinity|fresh(?!est)|freshwater|river|runoff", "sss", "<", 33.8),
    (r"high[- ]salinity|saline|salty|evaporation", "sss", ">=", 36.0),
    (r"warm|warm pool|hot", "sst", ">=", 28.0),
    (r"cold|cool", "sst", "<=", 22.0),
    (r"shallow mixed|shallow mld", "mld_dbar", "<=", 25.0),
    (r"deep mixed|deep mld", "mld_dbar", ">=", 60.0),
]


def facets(question: str) -> list[tuple[str, str, float]]:
    import re
    t = question.lower()
    return [(col, op, val) for rx, col, op, val in FACETS if re.search(rx, t)]
