from __future__ import annotations

import difflib
import re
import unicodedata
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# File paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "split_xgb_batted_distance.joblib"
PARK_FENCE_FILE = BASE_DIR / "park_fence_data_28parks_5zones_hr.csv"


# ============================================================
# Constants
# ============================================================

NUMERIC_FEATURES = [
    "launch_speed",
    "launch_angle",
    "spray_angle",
    "R_ideal_ft",
    "attack_angle",
    "swing_launch_gap",
    "abs_swing_launch_gap",
]

CATEGORICAL_FEATURES = [
    "stand",
    "p_throws",
    "matchup_side",
    "spray_zone5",
    "batted_direction5",
    "launch_angle_bin",
]

SPRAY_REP_ANGLE = {
    "left_line": -35.0,
    "left_gap": -18.0,
    "center": 0.0,
    "right_gap": 18.0,
    "right_line": 35.0,
}

SPRAY_OPTIONS = {
    "좌측 라인 / Left line": "left_line",
    "좌중간 / Left gap": "left_gap",
    "중앙 / Center": "center",
    "우중간 / Right gap": "right_gap",
    "우측 라인 / Right line": "right_line",
}

PITCHER_HAND_OPTIONS = {
    "우완 투수 / RHP": "R",
    "좌완 투수 / LHP": "L",
}


# ============================================================
# Text utilities
# ============================================================

def display_player_name(raw_name: str) -> str:
    """'Judge, Aaron' -> 'Aaron Judge'."""
    s = str(raw_name).strip()
    if "," in s:
        last, first = [p.strip() for p in s.split(",", 1)]
        if first and last:
            return f"{first} {last}"
    return s


def canonical_player_key(name: str) -> str:
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""

    s = str(name).strip()
    if not s:
        return ""

    if "," in s:
        parts = [p.strip() for p in s.split(",", 1)]
        if len(parts) == 2 and parts[0] and parts[1]:
            s = f"{parts[1]} {parts[0]}"

    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_text_for_search(text: str) -> str:
    return canonical_player_key(text)


# ============================================================
# Feature engineering
# ============================================================

def normalize_spray_zone5(zone: str) -> str:
    z = str(zone).strip().lower().replace(" ", "_").replace("-", "_")

    aliases = {
        "left_line": {"left_line", "ll", "좌측라인", "좌_라인", "좌라인", "좌선상", "leftfield_line"},
        "left_gap": {"left_gap", "lg", "left", "l", "좌중간", "좌측", "좌", "left_center"},
        "center": {"center", "c", "middle", "중", "중앙", "중견", "center_field"},
        "right_gap": {"right_gap", "rg", "right", "r", "우중간", "우측", "우", "right_center"},
        "right_line": {"right_line", "rl", "우측라인", "우_라인", "우라인", "우선상", "rightfield_line"},
    }

    for key, vals in aliases.items():
        if z in vals:
            return key

    raise ValueError("타구 방향은 left_line, left_gap, center, right_gap, right_line 중 하나여야 합니다.")


def spray_zone5_to_angle(zone: str) -> float:
    return SPRAY_REP_ANGLE[normalize_spray_zone5(zone)]


def normalize_stand(stand: str) -> str:
    s = str(stand).strip().upper()
    if s in ["R", "RIGHT", "우", "우타"]:
        return "R"
    if s in ["L", "LEFT", "좌", "좌타"]:
        return "L"
    raise ValueError("stand는 R 또는 L이어야 합니다.")


def normalize_pitcher_hand(p_throws: str) -> str:
    s = str(p_throws).strip().upper()
    if s in ["R", "RIGHT", "우", "우완"]:
        return "R"
    if s in ["L", "LEFT", "좌", "좌완"]:
        return "L"
    if s in ["", "NAN", "NONE", "UNKNOWN"]:
        return "unknown"
    raise ValueError("Throwing hand는 R 또는 L이어야 합니다.")


def infer_matchup_side(stand: str, p_throws: str) -> str:
    st = normalize_stand(stand)
    ph = normalize_pitcher_hand(p_throws)
    if ph not in ["R", "L"]:
        return "unknown"
    return "same_hand" if st == ph else "opposite_hand"


def classify_batted_direction5(spray_zone5: str, stand: str) -> str:
    zone = normalize_spray_zone5(spray_zone5)
    st = normalize_stand(stand)

    if zone == "center":
        return "center"

    if st == "R":
        mapping = {
            "left_line": "pull_line",
            "left_gap": "pull_gap",
            "right_gap": "oppo_gap",
            "right_line": "oppo_line",
        }
    else:
        mapping = {
            "right_line": "pull_line",
            "right_gap": "pull_gap",
            "left_gap": "oppo_gap",
            "left_line": "oppo_line",
        }

    return mapping[zone]


def infer_model_type_from_launch_angle(launch_angle: float) -> str:
    la = float(launch_angle)
    if 10 <= la < 25:
        return "line_drive"
    if 25 <= la <= 50:
        return "fly_ball"
    return "out_of_scope"


def launch_angle_bin(launch_angle: float) -> str:
    la = float(launch_angle)
    if 10 <= la < 15:
        return "la_10_15"
    if 15 <= la < 20:
        return "la_15_20"
    if 20 <= la < 25:
        return "la_20_25"
    if 25 <= la < 30:
        return "la_25_30"
    if 30 <= la < 35:
        return "la_30_35"
    if 35 <= la < 40:
        return "la_35_40"
    if 40 <= la <= 50:
        return "la_40_50"
    return "la_out_of_scope"


def calc_r_ideal_ft(launch_speed_mph, launch_angle_deg, z0_m: float = 1.0) -> np.ndarray:
    g = 9.81
    v0 = np.asarray(launch_speed_mph, dtype=float) * 0.44704
    theta = np.radians(np.asarray(launch_angle_deg, dtype=float))

    vx = v0 * np.cos(theta)
    vz = v0 * np.sin(theta)

    t_flight = (vz + np.sqrt(np.maximum(vz ** 2 + 2 * g * z0_m, 0))) / g
    r_m = vx * t_flight
    return r_m * 3.28084


def add_model_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["stand"] = df["stand"].apply(normalize_stand)
    df["p_throws"] = df["p_throws"].apply(normalize_pitcher_hand)
    df["matchup_side"] = [infer_matchup_side(s, p) for s, p in zip(df["stand"], df["p_throws"])]

    df["spray_zone5"] = df["spray_zone5"].apply(normalize_spray_zone5)
    df["spray_angle"] = df["spray_zone5"].apply(spray_zone5_to_angle)
    df["batted_direction5"] = [
        classify_batted_direction5(z, s) for z, s in zip(df["spray_zone5"], df["stand"])
    ]

    df["model_type"] = df["launch_angle"].apply(infer_model_type_from_launch_angle)
    df["launch_angle_bin"] = df["launch_angle"].apply(launch_angle_bin)
    df["R_ideal_ft"] = calc_r_ideal_ft(df["launch_speed"], df["launch_angle"])

    df["attack_angle"] = pd.to_numeric(df["attack_angle"], errors="coerce")
    df["swing_launch_gap"] = df["launch_angle"].astype(float) - df["attack_angle"].astype(float)
    df["abs_swing_launch_gap"] = df["swing_launch_gap"].abs()

    return df


def prepare_prediction_matrix(df: pd.DataFrame, feature_columns: list[str], medians: pd.Series) -> pd.DataFrame:
    x_num = df[NUMERIC_FEATURES].copy().fillna(medians)

    x_cat = df[CATEGORICAL_FEATURES].copy().fillna("missing").astype(str)
    x_cat = pd.get_dummies(x_cat, columns=CATEGORICAL_FEATURES)

    x = pd.concat([x_num.reset_index(drop=True), x_cat.reset_index(drop=True)], axis=1)
    x = x.reindex(columns=feature_columns, fill_value=0)
    return x


# ============================================================
# Player lookup
# ============================================================

def lookup_player_row(player_profiles: pd.DataFrame, player_name: str) -> Optional[pd.Series]:
    if player_profiles is None or player_profiles.empty:
        return None

    key = canonical_player_key(player_name)
    if not key:
        return None

    keys = player_profiles["player_key"].astype(str).tolist()

    if key in set(keys):
        match_key = key
    else:
        close = difflib.get_close_matches(key, keys, n=1, cutoff=0.78)
        if not close:
            return None
        match_key = close[0]

    return player_profiles[player_profiles["player_key"] == match_key].iloc[0]


def resolve_stand_for_single(
    player: str,
    p_throws: str,
    player_profiles: pd.DataFrame,
) -> Tuple[str, str, str, int, bool]:
    row = lookup_player_row(player_profiles, player)
    if row is None:
        raise ValueError(f"선수 '{player}'를 모델의 player profile에서 찾지 못했습니다.")

    matched_name = str(row["player_name_sample"])
    n = int(row.get("player_attack_angle_n", 0))
    is_switch = bool(row.get("is_switch_profile", False))

    if is_switch:
        ph = normalize_pitcher_hand(p_throws)
        col = "stand_vs_RHP" if ph == "R" else "stand_vs_LHP"
        if pd.notna(row.get(col, np.nan)):
            return normalize_stand(row[col]), f"player_profile_vs_{ph}HP", matched_name, n, is_switch

        return normalize_stand(row["primary_stand"]), "player_primary_stand_fallback", matched_name, n, is_switch

    return normalize_stand(row["primary_stand"]), "player_primary_stand", matched_name, n, is_switch


def lookup_player_attack_angle(
    player_profiles: pd.DataFrame,
    player_stand_profiles: pd.DataFrame,
    player_name: str,
    stand: str,
    league_attack_angle: float,
    min_player_bbe: int = 5,
) -> Tuple[float, str, int, str]:
    row = lookup_player_row(player_profiles, player_name)
    if row is None:
        return float(league_attack_angle), "league_average", 0, "league_fallback_not_found"

    player_key = str(row["player_key"])
    matched_name = str(row["player_name_sample"])
    st = normalize_stand(stand)

    ps = player_stand_profiles[
        (player_stand_profiles["player_key"].astype(str) == player_key)
        & (player_stand_profiles["stand"].astype(str) == st)
    ]

    if not ps.empty:
        r = ps.iloc[0]
        n = int(r.get("player_stand_attack_angle_n", 0))
        source = "player_stand_shrunk" if n >= int(min_player_bbe) else "player_stand_shrunk_low_n"
        return float(r["player_stand_attack_angle_shrunk"]), matched_name, n, source

    n = int(row.get("player_attack_angle_n", 0))
    source = "player_overall_shrunk_no_stand_profile" if n >= int(min_player_bbe) else "player_overall_shrunk_low_n"
    return float(row["player_attack_angle_shrunk"]), matched_name, n, source


def build_player_options(player_profiles: pd.DataFrame) -> pd.DataFrame:
    if player_profiles is None or player_profiles.empty:
        return pd.DataFrame(
            columns=["display_name", "player_key", "player_name_sample", "search_key", "n", "is_switch"]
        )

    out = player_profiles.copy()
    out["display_name"] = out["player_name_sample"].apply(display_player_name)
    out["search_key"] = out["display_name"].apply(normalize_text_for_search)
    out["n"] = out.get("player_attack_angle_n", 0).fillna(0).astype(int)
    out["is_switch"] = out.get("is_switch_profile", False).fillna(False).astype(bool)
    out = out.sort_values(["display_name", "n"], ascending=[True, False])

    return out[["display_name", "player_key", "player_name_sample", "search_key", "n", "is_switch"]].reset_index(drop=True)


def filter_player_candidates(options: pd.DataFrame, query: str, limit: int = 25) -> list[str]:
    if options.empty:
        return []

    q = normalize_text_for_search(query)
    if not q:
        top = options.sort_values("n", ascending=False).head(limit)
        return top["display_name"].tolist()

    mask = options["search_key"].str.contains(re.escape(q), na=False)
    candidates = options[mask].copy()

    if candidates.empty:
        q_key = canonical_player_key(query)
        mask2 = options["player_key"].astype(str).str.contains(re.escape(q_key), na=False)
        candidates = options[mask2].copy()

    candidates = candidates.sort_values("n", ascending=False).head(limit)
    return candidates["display_name"].tolist()


def get_selected_raw_player_name(options: pd.DataFrame, display_name: str) -> str:
    row = options[options["display_name"] == display_name]
    if row.empty:
        return display_name
    return str(row.iloc[0]["player_name_sample"])


def get_profile_summary(player_profiles: pd.DataFrame, selected_raw_name: str) -> Optional[pd.Series]:
    key = canonical_player_key(selected_raw_name)
    if not key:
        return None

    rows = player_profiles[player_profiles["player_key"].astype(str) == key]
    if rows.empty:
        return lookup_player_row(player_profiles, selected_raw_name)

    return rows.iloc[0]


# ============================================================
# Distance prediction
# ============================================================

def make_single_input(
    ev: float,
    la: float,
    spray: str,
    stand: str,
    p_throws: str,
    attack_angle: float,
) -> pd.DataFrame:
    row = pd.DataFrame(
        [
            {
                "launch_speed": float(ev),
                "launch_angle": float(la),
                "spray_zone5": normalize_spray_zone5(spray),
                "stand": normalize_stand(stand),
                "p_throws": normalize_pitcher_hand(p_throws),
                "attack_angle": float(attack_angle),
            }
        ]
    )
    return add_model_features(row)


def predict_single(
    model_bundles: dict,
    ev: float,
    la: float,
    spray: str,
    p_throws: str,
    player: str,
    player_profiles: pd.DataFrame,
    player_stand_profiles: pd.DataFrame,
    league_attack_angle: float,
    min_player_bbe: int = 5,
) -> Tuple[pd.DataFrame, float, dict]:
    resolved_stand, stand_source, matched_stand_name, stand_n, is_switch = resolve_stand_for_single(
        player, p_throws, player_profiles
    )

    resolved_attack_angle, matched_aa_name, aa_n, aa_source = lookup_player_attack_angle(
        player_profiles,
        player_stand_profiles,
        player,
        resolved_stand,
        league_attack_angle,
        min_player_bbe=min_player_bbe,
    )

    single = make_single_input(ev, la, spray, resolved_stand, p_throws, resolved_attack_angle)

    model_type = single.loc[0, "model_type"]
    if model_type not in model_bundles:
        raise ValueError("이 발사각은 모델 학습 범위 밖입니다. 10~50도 범위를 권장합니다.")

    bundle = model_bundles[model_type]
    x_single = prepare_prediction_matrix(single, bundle["feature_columns"], bundle["medians"])
    pred = float(bundle["model"].predict(x_single)[0])

    single["stand_source"] = stand_source
    single["attack_angle_source"] = aa_source
    single["matched_player_name"] = matched_aa_name if matched_aa_name != "league_average" else matched_stand_name
    single["matched_player_attack_angle_n"] = aa_n
    single["matched_player_stand_n"] = stand_n
    single["is_switch_profile"] = is_switch

    info = {
        "stand": resolved_stand,
        "stand_source": stand_source,
        "p_throws": single.loc[0, "p_throws"],
        "matchup_side": single.loc[0, "matchup_side"],
        "is_switch_profile": is_switch,
        "attack_angle": resolved_attack_angle,
        "attack_angle_source": aa_source,
        "matched_player_name": single.loc[0, "matched_player_name"],
        "matched_player_attack_angle_n": aa_n,
        "matched_player_stand_n": stand_n,
    }

    return single, pred, info


def make_warning(ev: float, la: float, model_type: str) -> Optional[str]:
    if model_type == "line_drive" and ev >= 100 and 10 <= la <= 20:
        return "고속·저발사각 라인드라이브 구간입니다. 실제 스핀, 수비 개입, projected distance 영향으로 오차가 커질 수 있습니다."
    if la < 10 or la > 50:
        return "현재 모델의 권장 발사각 범위는 10~50도입니다."
    return None


# ============================================================
# Home run detector
# ============================================================

def load_park_fence_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["park", "spray_zone5", "fence_distance_ft", "fence_height_ft"])

    df = pd.read_csv(path)
    df["spray_zone5"] = df["spray_zone5"].apply(normalize_spray_zone5)
    df["fence_distance_ft"] = pd.to_numeric(df["fence_distance_ft"], errors="coerce")
    df["fence_height_ft"] = pd.to_numeric(df["fence_height_ft"], errors="coerce")

    if "usable_for_hr_detector" in df.columns:
        df = df[df["usable_for_hr_detector"].astype(str).str.lower().isin(["yes", "true", "1"])].copy()

    df = df.dropna(subset=["park", "spray_zone5", "fence_distance_ft", "fence_height_ft"])
    return df


def simulate_late_drop_trajectory_ft(
    launch_speed_mph: float,
    launch_angle_deg: float,
    initial_lift_coef: float,
    drag_coef: float = 0.35,
    air_density: float = 1.225,
    initial_height_ft: float = 3.0,
    lift_decay_start_s: float = 1.8,
    lift_decay_tau_s: float = 0.9,
    dt: float = 0.003,
    max_time: float = 12.0,
) -> pd.DataFrame:
    """
    2D baseball trajectory with drag and time-decaying lift.

    Goal:
    - Early flight: enough lift/carry to look like the ball is still "riding"
    - Late flight: lift decays, horizontal speed has dropped, so dz/dx becomes steep
      and the ball falls more sharply.

    This is not a full Statcast 3D trajectory.
    It is a practical HR-detector trajectory reconstructed from EV, LA, and predicted distance.
    """
    mass = 0.145       # kg
    radius = 0.0366    # m
    area = np.pi * radius ** 2
    g = 9.80665

    theta = np.radians(float(launch_angle_deg))
    v0 = float(launch_speed_mph) * 0.44704

    x = 0.0
    z = float(initial_height_ft) * 0.3048
    vx = v0 * np.cos(theta)
    vz = v0 * np.sin(theta)

    rows = []
    t = 0.0
    k = 0.5 * float(air_density) * area / mass

    while t <= max_time and x >= -5 and z >= -10:
        # Store effective lift too, so we can inspect the late-drop behavior.
        if t <= float(lift_decay_start_s):
            lift_eff = float(initial_lift_coef)
        else:
            lift_eff = float(initial_lift_coef) * np.exp(
                -(t - float(lift_decay_start_s)) / max(float(lift_decay_tau_s), 1e-6)
            )

        rows.append((t, x * 3.28084, z * 3.28084, vx, vz, lift_eff))

        v = max(np.hypot(vx, vz), 1e-9)

        # Drag: opposite to velocity
        ax_drag = -k * float(drag_coef) * v * vx
        az_drag = -k * float(drag_coef) * v * vz

        # Lift: perpendicular to velocity, upward for positive lift_eff
        ax_lift = -k * lift_eff * v * vz
        az_lift =  k * lift_eff * v * vx

        ax = ax_drag + ax_lift
        az = az_drag + az_lift - g

        vx += ax * dt
        vz += az * dt
        x += vx * dt
        z += vz * dt
        t += dt

        if t > 0.03 and z <= 0:
            rows.append((t, x * 3.28084, z * 3.28084, vx, vz, lift_eff))
            break

    return pd.DataFrame(
        rows,
        columns=["time_s", "x_ft", "z_ft", "vx_mps", "vz_mps", "effective_lift_coef"],
    )


def landing_distance_from_traj(traj: pd.DataFrame) -> float:
    if traj.empty:
        return np.nan

    z = traj["z_ft"].to_numpy()
    x = traj["x_ft"].to_numpy()

    for i in range(1, len(traj)):
        if z[i] <= 0 <= z[i - 1]:
            denom = z[i - 1] - z[i]
            if abs(denom) < 1e-9:
                return float(x[i])
            frac = z[i - 1] / denom
            return float(x[i - 1] + frac * (x[i] - x[i - 1]))

    return float(x[-1])


def height_at_x_from_traj(traj: pd.DataFrame, x_query_ft: float) -> float:
    if traj.empty:
        return np.nan

    xq = float(x_query_ft)
    x = traj["x_ft"].to_numpy()
    z = traj["z_ft"].to_numpy()

    if xq < x[0] or xq > np.nanmax(x):
        return np.nan

    return float(np.interp(xq, x, z))


def calibrate_initial_lift_to_predicted_distance(
    launch_speed_mph: float,
    launch_angle_deg: float,
    predicted_distance_ft: float,
    drag_coef: float = 0.35,
    air_density: float = 1.225,
    initial_height_ft: float = 3.0,
    lift_decay_start_s: float = 1.8,
    lift_decay_tau_s: float = 0.9,
    lift_min: float = -0.20,
    lift_max: float = 1.20,
    iterations: int = 30,
) -> tuple[float, pd.DataFrame, float, str]:
    """
    Calibrate the initial lift coefficient so the late-drop trajectory lands
    near the XGBoost-predicted distance.

    Difference from constant-lift model:
    - Lift is strong early.
    - After lift_decay_start_s, lift decays exponentially.
    - This creates a trajectory that carries early and drops faster late.
    """
    target = float(predicted_distance_ft)

    traj_lo = simulate_late_drop_trajectory_ft(
        launch_speed_mph, launch_angle_deg, lift_min, drag_coef, air_density,
        initial_height_ft, lift_decay_start_s, lift_decay_tau_s
    )
    traj_hi = simulate_late_drop_trajectory_ft(
        launch_speed_mph, launch_angle_deg, lift_max, drag_coef, air_density,
        initial_height_ft, lift_decay_start_s, lift_decay_tau_s
    )

    dist_lo = landing_distance_from_traj(traj_lo)
    dist_hi = landing_distance_from_traj(traj_hi)

    if not np.isfinite(dist_lo) or not np.isfinite(dist_hi):
        traj_mid = simulate_late_drop_trajectory_ft(
            launch_speed_mph, launch_angle_deg, 0.25, drag_coef, air_density,
            initial_height_ft, lift_decay_start_s, lift_decay_tau_s
        )
        return 0.25, traj_mid, landing_distance_from_traj(traj_mid), "fallback_default_late_drop"

    if target <= min(dist_lo, dist_hi):
        chosen_lift = lift_min if dist_lo <= dist_hi else lift_max
        chosen_traj = traj_lo if dist_lo <= dist_hi else traj_hi
        return chosen_lift, chosen_traj, landing_distance_from_traj(chosen_traj), "target_below_late_drop_range"

    if target >= max(dist_lo, dist_hi):
        chosen_lift = lift_max if dist_hi >= dist_lo else lift_min
        chosen_traj = traj_hi if dist_hi >= dist_lo else traj_lo
        return chosen_lift, chosen_traj, landing_distance_from_traj(chosen_traj), "target_above_late_drop_range"

    lo, hi = lift_min, lift_max
    best_lift = 0.25
    best_traj = None
    best_dist = np.nan
    increasing = dist_hi > dist_lo

    for _ in range(iterations):
        mid = (lo + hi) / 2.0
        traj_mid = simulate_late_drop_trajectory_ft(
            launch_speed_mph, launch_angle_deg, mid, drag_coef, air_density,
            initial_height_ft, lift_decay_start_s, lift_decay_tau_s
        )
        dist_mid = landing_distance_from_traj(traj_mid)

        best_lift, best_traj, best_dist = mid, traj_mid, dist_mid

        if increasing:
            if dist_mid < target:
                lo = mid
            else:
                hi = mid
        else:
            if dist_mid < target:
                hi = mid
            else:
                lo = mid

    return float(best_lift), best_traj, float(best_dist), "calibrated_late_drop_to_predicted_distance"


def judge_home_run(
    predicted_distance_ft: float,
    launch_speed_mph: float,
    launch_angle_deg: float,
    fence_distance_ft: float,
    fence_height_ft: float,
    margin_ft: float = 2.0,
    initial_height_ft: float = 3.0,
    drag_coef: float = 0.35,
    air_density: float = 1.225,
    lift_decay_start_s: float = 1.8,
    lift_decay_tau_s: float = 0.9,
) -> dict:
    R = float(predicted_distance_ft)
    D = float(fence_distance_ft)
    H = float(fence_height_ft)

    initial_lift_coef, traj, aero_landing_ft, calibration_status = calibrate_initial_lift_to_predicted_distance(
        launch_speed_mph=launch_speed_mph,
        launch_angle_deg=launch_angle_deg,
        predicted_distance_ft=R,
        drag_coef=drag_coef,
        air_density=air_density,
        initial_height_ft=initial_height_ft,
        lift_decay_start_s=lift_decay_start_s,
        lift_decay_tau_s=lift_decay_tau_s,
    )

    z_wall = height_at_x_from_traj(traj, D)
    if not np.isfinite(z_wall):
        z_wall = -999.0

    clearance = z_wall - H

    if R < D:
        result = "NOT_HR"
        reason = "예측 비거리가 펜스 거리보다 짧습니다."
    elif clearance >= margin_ft:
        result = "HOME_RUN"
        reason = f"late-drop 궤적 기준, 펜스 지점에서 공이 담장보다 {clearance:.1f} ft 높습니다."
    elif clearance >= 0:
        result = "BORDERLINE_HR"
        reason = f"late-drop 궤적 기준, 담장은 넘지만 여유가 {clearance:.1f} ft로 작습니다."
    elif clearance >= -margin_ft:
        result = "BORDERLINE_NOT_HR"
        reason = f"late-drop 궤적 기준, 펜스 근처에서 {abs(clearance):.1f} ft 부족합니다."
    else:
        result = "NOT_HR"
        reason = f"late-drop 궤적 기준, 펜스 지점에서 {abs(clearance):.1f} ft 부족합니다."

    return {
        "result": result,
        "ball_height_at_wall_ft": z_wall,
        "clearance_ft": clearance,
        "reason": reason,
        "initial_lift_coef": initial_lift_coef,
        "effective_lift_coef": initial_lift_coef,
        "aero_landing_distance_ft": aero_landing_ft,
        "trajectory_calibration": calibration_status,
        "lift_decay_start_s": lift_decay_start_s,
        "lift_decay_tau_s": lift_decay_tau_s,
    }


def judge_for_park(
    park_df: pd.DataFrame,
    park: str,
    spray_zone5: str,
    predicted_distance_ft: float,
    launch_angle_deg: float,
    margin_ft: float,
    initial_height_ft: float,
    launch_speed_mph: float = 100.0,
    drag_coef: float = 0.35,
    air_density: float = 1.225,
    lift_decay_start_s: float = 1.8,
    lift_decay_tau_s: float = 0.9,
) -> dict:
    zone = normalize_spray_zone5(spray_zone5)
    rows = park_df[(park_df["park"] == park) & (park_df["spray_zone5"] == zone)]

    if rows.empty:
        raise ValueError(f"{park} / {zone} 방향의 펜스 데이터를 찾지 못했습니다.")

    r = rows.iloc[0]
    fence_distance = float(r["fence_distance_ft"])
    fence_height = float(r["fence_height_ft"])

    out = judge_home_run(
        predicted_distance_ft=predicted_distance_ft,
        launch_speed_mph=launch_speed_mph,
        launch_angle_deg=launch_angle_deg,
        fence_distance_ft=fence_distance,
        fence_height_ft=fence_height,
        margin_ft=margin_ft,
        initial_height_ft=initial_height_ft,
        drag_coef=drag_coef,
        air_density=air_density,
        lift_decay_start_s=lift_decay_start_s,
        lift_decay_tau_s=lift_decay_tau_s,
    )
    out.update(
        {
            "park": park,
            "spray_zone5": zone,
            "fence_distance_ft": fence_distance,
            "fence_height_ft": fence_height,
        }
    )
    return out


def judge_all_parks(
    park_df: pd.DataFrame,
    spray_zone5: str,
    predicted_distance_ft: float,
    launch_angle_deg: float,
    margin_ft: float,
    initial_height_ft: float,
    launch_speed_mph: float = 100.0,
    drag_coef: float = 0.35,
    air_density: float = 1.225,
    lift_decay_start_s: float = 1.8,
    lift_decay_tau_s: float = 0.9,
) -> pd.DataFrame:
    zone = normalize_spray_zone5(spray_zone5)
    rows = park_df[park_df["spray_zone5"] == zone].copy()
    records = []

    for _, r in rows.iterrows():
        res = judge_home_run(
            predicted_distance_ft=predicted_distance_ft,
            launch_speed_mph=launch_speed_mph,
            launch_angle_deg=launch_angle_deg,
            fence_distance_ft=float(r["fence_distance_ft"]),
            fence_height_ft=float(r["fence_height_ft"]),
            margin_ft=margin_ft,
            initial_height_ft=initial_height_ft,
            drag_coef=drag_coef,
            air_density=air_density,
            lift_decay_start_s=lift_decay_start_s,
            lift_decay_tau_s=lift_decay_tau_s,
        )
        records.append(
            {
                "park": r["park"],
                "spray_zone5": zone,
                "fence_distance_ft": float(r["fence_distance_ft"]),
                "fence_height_ft": float(r["fence_height_ft"]),
                "ball_height_at_wall_ft": res["ball_height_at_wall_ft"],
                "clearance_ft": res["clearance_ft"],
                "result": res["result"],
                "initial_lift_coef": res["initial_lift_coef"],
                "aero_landing_distance_ft": res["aero_landing_distance_ft"],
                "trajectory_calibration": res["trajectory_calibration"],
                "reason": res["reason"],
            }
        )

    out = pd.DataFrame(records)
    order = {"HOME_RUN": 0, "BORDERLINE_HR": 1, "BORDERLINE_NOT_HR": 2, "NOT_HR": 3}
    out["_order"] = out["result"].map(order).fillna(9)
    out = out.sort_values(["_order", "clearance_ft"], ascending=[True, False]).drop(columns="_order")
    return out.reset_index(drop=True)


# ============================================================
# Load files
# ============================================================

st.set_page_config(
    page_title="Batted Ball HR Simulator",
    page_icon="⚾",
    layout="wide",
)

st.title("⚾ Batted Ball Distance & Home Run Simulator")
st.caption("XGBoost 비거리 예측 + late-drop 궤적 기반 홈런 판독")

if not MODEL_FILE.exists():
    st.error("모델 파일을 찾을 수 없습니다.")
    st.code(str(MODEL_FILE))
    st.info("GitHub 저장소에서 app.py와 split_xgb_batted_distance.joblib가 같은 위치에 있는지 확인하세요.")
    st.stop()

try:
    bundle = joblib.load(MODEL_FILE)
except Exception as exc:
    st.error("모델 파일을 불러오지 못했습니다.")
    st.code(str(exc))
    st.stop()

try:
    models = bundle["models"]
    player_profiles = bundle["player_profiles"]
    player_stand_profiles = bundle["player_stand_profiles"]
    league_attack_angle = float(bundle["league_attack_angle_mean"])
except KeyError as exc:
    st.error("모델 파일 구조가 현재 app.py와 맞지 않습니다.")
    st.code(f"누락된 key: {exc}")
    st.stop()

park_fence_data = load_park_fence_data(PARK_FENCE_FILE)
if park_fence_data.empty:
    st.error("구장 펜스 CSV를 찾지 못했거나 사용할 수 있는 데이터가 없습니다.")
    st.code(str(PARK_FENCE_FILE))
    st.info("app.py와 park_fence_data_28parks_5zones_hr.csv를 같은 위치에 두세요.")
    st.stop()

player_options = build_player_options(player_profiles)
park_options = sorted(park_fence_data["park"].unique().tolist())


# ============================================================
# Main UI
# ============================================================

left, right = st.columns([1.1, 0.9])

with left:
    st.subheader("입력")

    player_query = st.text_input(
        "1. 선수 이름 검색",
        value="",
        placeholder="예: Aaron Judge",
        help="입력하면 아래 후보 목록이 줄어듭니다.",
    )

    candidates = filter_player_candidates(player_options, player_query, limit=30)

    if not candidates:
        st.warning("검색 후보가 없습니다. 철자를 확인하거나 다른 이름으로 입력해보세요.")
        selected_player_display = None
    else:
        selected_player_display = st.selectbox(
            "선수 후보",
            options=candidates,
            index=0,
            help="후보 목록에서 실제 선수를 선택하세요.",
        )

    p_throws_label = st.selectbox(
        "2. Throwing hand",
        options=list(PITCHER_HAND_OPTIONS.keys()),
        index=0,
        help="투수 손입니다. 스위치히터의 batting stand 결정에 사용됩니다.",
    )
    p_throws = PITCHER_HAND_OPTIONS[p_throws_label]

    col_ev, col_la = st.columns(2)
    with col_ev:
        ev = st.number_input(
            "3. Exit Velocity (mph)",
            min_value=30.0,
            max_value=125.0,
            value=100.0,
            step=0.1,
        )
    with col_la:
        la = st.number_input(
            "4. Launch Angle (deg)",
            min_value=10.0,
            max_value=50.0,
            value=30.0,
            step=0.1,
        )

    spray_label = st.selectbox(
        "5. 타구 방향",
        options=list(SPRAY_OPTIONS.keys()),
        index=0,
    )
    spray = SPRAY_OPTIONS[spray_label]

    st.subheader("홈런 판독 설정")

    park = st.selectbox(
        "6. 구장",
        options=park_options,
        index=0,
        help="Coors Field와 Oracle Park는 제외된 CSV 기준입니다.",
    )

    compare_all_parks = st.checkbox("같은 타구를 전체 구장에서 비교", value=True)

    col_margin, col_z0 = st.columns(2)
    with col_margin:
        margin_ft = st.number_input(
            "Borderline margin (ft)",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.5,
            help="펜스를 몇 ft 이상 넘으면 확실한 홈런으로 볼지 정하는 여유값입니다.",
        )
    with col_z0:
        initial_height_ft = st.number_input(
            "초기 타구 높이 z0 (ft)",
            min_value=1.0,
            max_value=6.0,
            value=3.0,
            step=0.5,
            help="타격 순간 공의 높이입니다. 공기역학 궤적 복원용 값입니다.",
        )

    with st.expander("공기역학 설정"):
        drag_coef = st.slider(
            "Drag coefficient Cd",
            min_value=0.25,
            max_value=0.50,
            value=0.35,
            step=0.01,
            help="공기저항 계수입니다. 값이 클수록 공이 더 빨리 감속합니다.",
        )
        air_density = st.slider(
            "Air density rho (kg/m³)",
            min_value=0.90,
            max_value=1.30,
            value=1.225,
            step=0.005,
            help="공기 밀도입니다. 고지대/더운 날씨는 더 낮은 값을 쓸 수 있습니다.",
        )
        lift_decay_start_s = st.slider(
            "Late-drop start time (s)",
            min_value=0.8,
            max_value=3.5,
            value=1.8,
            step=0.1,
            help="이 시간 이후부터 lift를 감소시켜 후반부 낙하를 더 강하게 만듭니다.",
        )
        lift_decay_tau_s = st.slider(
            "Late-drop decay tau (s)",
            min_value=0.3,
            max_value=2.5,
            value=0.9,
            step=0.1,
            help="값이 작을수록 lift가 더 빨리 줄어들어 후반부에 더 급격히 떨어집니다.",
        )
        st.caption("초기 lift는 XGBoost 예측 비거리에 맞도록 자동 보정하고, 비행 후반에는 lift를 감소시켜 late-drop을 반영합니다.")

    predict_clicked = st.button("예측 실행", type="primary", use_container_width=True)


with right:
    st.subheader("선수 프로필")

    if selected_player_display is None:
        st.info("선수를 선택하면 profile이 표시됩니다.")
    else:
        selected_raw_name = get_selected_raw_player_name(player_options, selected_player_display)
        row = get_profile_summary(player_profiles, selected_raw_name)

        if row is None:
            st.warning("선수 profile을 찾지 못했습니다.")
        else:
            profile_table = pd.DataFrame(
                [
                    ["선수", selected_player_display],
                    ["표본 수", int(row.get("player_attack_angle_n", 0))],
                    ["Primary stand", str(row.get("primary_stand", "unknown"))],
                    ["Switch profile", bool(row.get("is_switch_profile", False))],
                    ["Attack angle mean", f"{float(row.get('player_attack_angle_mean', 0)):.2f}°"],
                    ["Attack angle shrunk", f"{float(row.get('player_attack_angle_shrunk', 0)):.2f}°"],
                    ["Stand vs RHP", str(row.get("stand_vs_RHP", "-"))],
                    ["Stand vs LHP", str(row.get("stand_vs_LHP", "-"))],
                ],
                columns=["항목", "값"],
            ).astype(str)

            st.dataframe(profile_table, hide_index=True, use_container_width=True)

    st.subheader("선택 구장 펜스")
    fence_preview = park_fence_data[park_fence_data["park"] == park][
        ["spray_zone5", "fence_distance_ft", "fence_height_ft"]
    ].copy()
    fence_preview = fence_preview.sort_values("spray_zone5")
    st.dataframe(fence_preview, hide_index=True, use_container_width=True)


# ============================================================
# Prediction output
# ============================================================

if predict_clicked:
    if selected_player_display is None:
        st.error("선수를 먼저 선택하세요.")
        st.stop()

    selected_raw_name = get_selected_raw_player_name(player_options, selected_player_display)

    try:
        single_df, pred_distance, info = predict_single(
            model_bundles=models,
            ev=float(ev),
            la=float(la),
            spray=spray,
            p_throws=p_throws,
            player=selected_raw_name,
            player_profiles=player_profiles,
            player_stand_profiles=player_stand_profiles,
            league_attack_angle=league_attack_angle,
            min_player_bbe=5,
        )
    except Exception as exc:
        st.error("예측에 실패했습니다.")
        st.code(str(exc))
        st.stop()

    row = single_df.iloc[0]
    warning = make_warning(float(ev), float(la), str(row["model_type"]))

    try:
        hr_result = judge_for_park(
            park_df=park_fence_data,
            park=park,
            spray_zone5=spray,
            predicted_distance_ft=pred_distance,
            launch_angle_deg=float(la),
            margin_ft=float(margin_ft),
            initial_height_ft=float(initial_height_ft),
            launch_speed_mph=float(ev),
            drag_coef=float(drag_coef),
            air_density=float(air_density),
            lift_decay_start_s=float(lift_decay_start_s),
            lift_decay_tau_s=float(lift_decay_tau_s),
        )
    except Exception as exc:
        st.error("홈런 판독에 실패했습니다.")
        st.code(str(exc))
        st.stop()

    st.divider()
    st.subheader("예측 결과")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted distance", f"{pred_distance:.1f} ft")
    m2.metric("Model type", str(row["model_type"]))
    m3.metric("Swing-launch gap", f"{float(row['swing_launch_gap']):.2f}°")
    m4.metric("HR result", hr_result["result"])

    if warning:
        st.warning(warning)

    if hr_result["result"] in ["HOME_RUN", "BORDERLINE_HR"]:
        st.success(hr_result["reason"])
    elif hr_result["result"] == "BORDERLINE_NOT_HR":
        st.warning(hr_result["reason"])
    else:
        st.info(hr_result["reason"])

    result_table = pd.DataFrame(
        [
            ["선수", selected_player_display],
            ["투수 손", p_throws],
            ["Resolved batting stand", str(row["stand"])],
            ["Matchup side", str(row["matchup_side"])],
            ["Exit Velocity", f"{float(ev):.1f} mph"],
            ["Launch Angle", f"{float(la):.1f}°"],
            ["Spray zone", str(row["spray_zone5"])],
            ["Batted direction", str(row["batted_direction5"])],
            ["Applied attack angle", f"{float(row['attack_angle']):.2f}°"],
            ["Attack angle source", str(info.get("attack_angle_source", "-"))],
            ["Stand source", str(info.get("stand_source", "-"))],
            ["Player attack angle sample n", int(info.get("matched_player_attack_angle_n", 0))],
            ["R_ideal_ft", f"{float(row['R_ideal_ft']):.1f} ft"],
            ["구장", park],
            ["Fence distance", f"{float(hr_result['fence_distance_ft']):.1f} ft"],
            ["Fence height", f"{float(hr_result['fence_height_ft']):.1f} ft"],
            ["Ball height at fence", f"{float(hr_result['ball_height_at_wall_ft']):.1f} ft"],
            ["Clearance", f"{float(hr_result['clearance_ft']):.1f} ft"],
            ["Initial lift coefficient", f"{float(hr_result['initial_lift_coef']):.3f}"],
            ["Late-drop start", f"{float(hr_result['lift_decay_start_s']):.1f} s"],
            ["Late-drop tau", f"{float(hr_result['lift_decay_tau_s']):.1f} s"],
            ["Aero landing distance", f"{float(hr_result['aero_landing_distance_ft']):.1f} ft"],
            ["Trajectory calibration", str(hr_result["trajectory_calibration"])],
            ["HR 판정", hr_result["result"]],
        ],
        columns=["항목", "값"],
    ).astype(str)

    st.dataframe(result_table, hide_index=True, use_container_width=True)

    if compare_all_parks:
        st.subheader("전체 구장 비교")

        all_results = judge_all_parks(
            park_df=park_fence_data,
            spray_zone5=spray,
            predicted_distance_ft=pred_distance,
            launch_angle_deg=float(la),
            margin_ft=float(margin_ft),
            initial_height_ft=float(initial_height_ft),
            launch_speed_mph=float(ev),
            drag_coef=float(drag_coef),
            air_density=float(air_density),
            lift_decay_start_s=float(lift_decay_start_s),
            lift_decay_tau_s=float(lift_decay_tau_s),
        )

        hr_count = int(all_results["result"].isin(["HOME_RUN", "BORDERLINE_HR"]).sum())
        total_count = int(len(all_results))
        st.caption(f"동일 타구 기준 홈런 또는 borderline HR: {hr_count} / {total_count}개 구장")

        show_cols = [
            "park",
            "result",
            "fence_distance_ft",
            "fence_height_ft",
            "ball_height_at_wall_ft",
            "clearance_ft",
            "initial_lift_coef",
            "trajectory_calibration",
            "reason",
        ]
        display_df = all_results[show_cols].copy()
        for c in ["fence_distance_ft", "fence_height_ft", "ball_height_at_wall_ft", "clearance_ft"]:
            display_df[c] = display_df[c].map(lambda x: f"{x:.1f}")
        if "initial_lift_coef" in display_df.columns:
            display_df["initial_lift_coef"] = display_df["initial_lift_coef"].map(lambda x: f"{x:.3f}")
        st.dataframe(display_df, hide_index=True, use_container_width=True)

    st.caption(
        "홈런 판정은 XGBoost 예측 비거리에 맞도록 초기 lift를 보정하고, "
        "비행 후반 lift를 감소시키는 late-drop 궤적으로 펜스 위치의 공 높이를 계산하는 후처리 모듈입니다."
    )
