# -*- coding: utf-8 -*-
"""Core analytical model for a small circular-loop antenna.

The calculations intentionally follow the approximate analytical model in the
user's desktop program. Material choice changes conductor loss, not the ideal
small-loop radiation resistance or normalized pattern in this model.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.special import j1

C0 = 299_792_458.0
MU0 = 4.0 * math.pi * 1e-7
SMALL_LOOP_LIMIT = 0.1
DEFAULT_FREQ_MHZ = 30.0
DEFAULT_WIRE_RADIUS_MM = 1.0
DEFAULT_TURNS = 1
DEFAULT_Z0_OHM = 50.0

# Resistivity values are approximate bulk values near 20 °C.
# Real wire values vary with purity, alloy, temperature and manufacturing.
MATERIALS: dict[str, dict[str, float | str]] = {
    "Đồng (Cu)": {
        "symbol": "Cu", "rho_ohm_m": 1.724e-8,
        "density_kg_m3": 8960.0, "color": "#C87533",
    },
    "Bạc (Ag)": {
        "symbol": "Ag", "rho_ohm_m": 1.59e-8,
        "density_kg_m3": 10490.0, "color": "#AEB8C2",
    },
    "Vàng (Au)": {
        "symbol": "Au", "rho_ohm_m": 2.44e-8,
        "density_kg_m3": 19320.0, "color": "#E3B341",
    },
    "Nhôm (Al)": {
        "symbol": "Al", "rho_ohm_m": 2.82e-8,
        "density_kg_m3": 2700.0, "color": "#95A3B3",
    },
}


def _positive_finite(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} phải là một số hợp lệ.") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} phải lớn hơn 0 và là số hữu hạn.")
    return result


def calculate_antenna(
    radius_cm: float,
    freq_mhz: float,
    wire_radius_mm: float = DEFAULT_WIRE_RADIUS_MM,
    n_turns: int = DEFAULT_TURNS,
    z0_ohm: float = DEFAULT_Z0_OHM,
    material: str = "Đồng (Cu)",
) -> dict[str, Any]:
    """Calculate approximate small-loop parameters for a selected conductor."""
    radius_cm = _positive_finite(radius_cm, "Bán kính vòng R")
    freq_mhz = _positive_finite(freq_mhz, "Tần số f")
    wire_radius_mm = _positive_finite(wire_radius_mm, "Bán kính dây dẫn a")
    z0_ohm = _positive_finite(z0_ohm, "Trở kháng tham chiếu Z0")

    if isinstance(n_turns, bool):
        raise ValueError("Số vòng N phải là số nguyên >= 1.")
    try:
        turns_float = float(n_turns)
    except (TypeError, ValueError) as exc:
        raise ValueError("Số vòng N phải là số nguyên >= 1.") from exc
    if not math.isfinite(turns_float) or turns_float < 1 or not turns_float.is_integer():
        raise ValueError("Số vòng N phải là số nguyên >= 1.")
    n_turns = int(turns_float)

    if material not in MATERIALS:
        raise ValueError(f"Vật liệu không được hỗ trợ: {material}")
    rho = float(MATERIALS[material]["rho_ohm_m"])
    density = float(MATERIALS[material]["density_kg_m3"])

    R = radius_cm / 100.0
    a = wire_radius_mm / 1000.0
    f = freq_mhz * 1e6
    omega = 2.0 * math.pi * f
    wavelength = C0 / f
    circumference = 2.0 * math.pi * R
    area = math.pi * R**2
    circumference_lambda = circumference / wavelength

    # Same analytical approximations as the supplied desktop program.
    radiation_resistance = 31_200.0 * ((n_turns * area) / wavelength**2) ** 2

    ratio = 8.0 * R / a
    if ratio <= math.exp(2.0):
        min_radius_cm = (math.exp(2.0) * wire_radius_mm) / 80.0
        raise ValueError(
            "Tỉ lệ kích thước dây/vòng không phù hợp với công thức điện cảm gần đúng. "
            f"Với a={wire_radius_mm:g} mm, hãy tăng R lớn hơn khoảng {min_radius_cm:.4g} cm."
        )
    inductance_single = MU0 * R * (math.log(ratio) - 2.0)
    inductance = (n_turns**2) * inductance_single

    wire_length = circumference * n_turns
    cross_section = math.pi * a**2
    resistance_dc = rho * wire_length / cross_section
    skin_depth = math.sqrt(2.0 * rho / (omega * MU0))
    if skin_depth < a:
        resistance_ac_skin = rho * wire_length / (2.0 * math.pi * a * skin_depth)
        resistance_ac = max(resistance_dc, resistance_ac_skin)
    else:
        resistance_ac = resistance_dc

    resistance_in = radiation_resistance + resistance_ac
    reactance_in = omega * inductance
    zin_abs = math.hypot(resistance_in, reactance_in)
    zin_phase = math.degrees(math.atan2(reactance_in, resistance_in))
    efficiency = radiation_resistance / (radiation_resistance + resistance_ac)
    directivity = 1.5
    gain_linear = directivity * efficiency
    gain_dbi = 10.0 * math.log10(gain_linear) if gain_linear > 0 else float("-inf")

    zin_complex = complex(resistance_in, reactance_in)
    gamma = abs((zin_complex - z0_ohm) / (zin_complex + z0_ohm))
    mismatch_efficiency = max(0.0, 1.0 - gamma**2)
    realized_gain_linear = gain_linear * mismatch_efficiency
    realized_gain_dbi = (
        10.0 * math.log10(realized_gain_linear)
        if realized_gain_linear > 0 else float("-inf")
    )
    return_loss = -20.0 * math.log10(max(gamma, 1e-15))
    swr = (1.0 + gamma) / max(1.0 - gamma, 1e-15)

    return {
        "radius_cm": radius_cm,
        "freq_mhz": freq_mhz,
        "wire_radius_mm": wire_radius_mm,
        "n_turns": n_turns,
        "z0_ohm": z0_ohm,
        "material": material,
        "material_symbol": MATERIALS[material]["symbol"],
        "rho_ohm_m": rho,
        "conductivity_s_m": 1.0 / rho,
        "density_kg_m3": density,
        "wavelength_m": wavelength,
        "circumference_m": circumference,
        "area_m2": area,
        "circumference_lambda": circumference_lambda,
        "is_small_loop": circumference_lambda < SMALL_LOOP_LIMIT,
        "Rr_ohm": radiation_resistance,
        "L_uH": inductance * 1e6,
        "skin_depth_um": skin_depth * 1e6,
        "Rdc_ohm": resistance_dc,
        "Rac_ohm": resistance_ac,
        "wire_length_m": wire_length,
        "wire_mass_g": density * wire_length * cross_section * 1000.0,
        "Zin_real_ohm": resistance_in,
        "Zin_imag_ohm": reactance_in,
        "Zin_magnitude_ohm": zin_abs,
        "Zin_phase_deg": zin_phase,
        "radiation_efficiency_pct": efficiency * 100.0,
        "directivity_linear": directivity,
        "directivity_dBi": 10.0 * math.log10(directivity),
        "gain_linear_approx": gain_linear,
        "gain_dBi_approx": gain_dbi,
        "gamma_z0_magnitude_approx": gamma,
        "mismatch_efficiency_z0_approx_pct": mismatch_efficiency * 100.0,
        "return_loss_z0_approx_db": return_loss,
        "swr_z0_approx": swr,
        "realized_gain_dBi_approx": realized_gain_dbi,
    }


def loop_field_pattern(theta_rad: np.ndarray | float, base: dict[str, Any]) -> np.ndarray:
    """Normalized |E(theta)| for the uniform-current circular-loop approximation."""
    theta = np.asarray(theta_rad, dtype=float)
    radius_m = float(base["radius_cm"]) / 100.0
    wavelength = float(base["wavelength_m"])
    k_radius = 2.0 * math.pi * radius_m / wavelength
    raw = np.abs(j1(k_radius * np.sin(theta)))
    peak = float(np.max(raw)) if raw.size else float(raw)
    if not math.isfinite(peak) or peak < 1e-14:
        raw = np.abs(np.sin(theta))
    else:
        raw = raw / peak
    peak2 = float(np.max(raw)) if np.size(raw) else float(raw)
    if peak2 > 0:
        raw = raw / peak2
    return np.clip(raw, 0.0, 1.0)


def make_sweep_data(
    base: dict[str, Any], n_r: int = 70, n_f: int = 55, heat_r_count: int = 42, heat_f_count: int = 34
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r0 = float(base["radius_cm"])
    f0 = float(base["freq_mhz"])
    awire = float(base["wire_radius_mm"])
    turns = int(base["n_turns"])
    z0 = float(base["z0_ohm"])
    material = str(base["material"])

    threshold_cm = math.exp(2.0) * awire / 80.0
    r_min = max(threshold_cm * 1.02, r0 * 0.20)
    r_min = min(r_min, r0)
    r_max = max(r0 * 2.0, r0 + 2.0, r_min * 1.5)
    radii = np.linspace(r_min, r_max, n_r)
    freqs = np.linspace(max(0.5, f0 * 0.30), f0 * 1.70, n_f)

    radius_rows = [calculate_antenna(float(r), f0, awire, turns, z0, material) for r in radii]
    freq_rows = [calculate_antenna(r0, float(f), awire, turns, z0, material) for f in freqs]

    heat_radii = np.linspace(r_min, r_max, heat_r_count)
    heat_freqs = np.linspace(max(0.5, f0 * 0.30), f0 * 1.70, heat_f_count)
    heat_rr_db = np.empty((len(heat_freqs), len(heat_radii)), dtype=float)
    heat_gain_db = np.empty_like(heat_rr_db)
    for i, freq in enumerate(heat_freqs):
        for j, radius in enumerate(heat_radii):
            result = calculate_antenna(float(radius), float(freq), awire, turns, z0, material)
            heat_rr_db[i, j] = 10.0 * math.log10(max(float(result["Rr_ohm"]), 1e-30))
            heat_gain_db[i, j] = float(result["gain_dBi_approx"])
    return radius_rows, freq_rows, heat_radii, heat_freqs, heat_rr_db, heat_gain_db


def compare_materials(
    radius_cm: float, freq_mhz: float, wire_radius_mm: float, n_turns: int, z0_ohm: float
) -> list[dict[str, Any]]:
    """Return same-geometry calculations for each supported conductor."""
    return [
        calculate_antenna(radius_cm, freq_mhz, wire_radius_mm, n_turns, z0_ohm, name)
        for name in MATERIALS
    ]
