"""
split_xgb_player_stand_pthrows_batted_distance.py

목적
- 사용자가 입력하기 쉬운 값에 attack_angle을 추가해 타구 비거리를 예측합니다.
- 기본 입력: 타구속도, 발사각, 스프레이 방향 5구간, 타자 방향, attack_angle
- Coors Field(COL), Oracle Park(SF)는 기본적으로 train/test에서 제외합니다.
- 발사각에 따라 line-drive 전용 모델과 fly-ball 전용 모델을 자동 선택합니다.
- attack_angle과 launch_angle의 차이로 swing_launch_gap을 생성합니다.

핵심 feature
1) launch_speed: 타구속도, mph
2) launch_angle: 발사각, degree
3) spray_zone5: left_line / left_gap / center / right_gap / right_line
4) stand: R / L
5) batted_direction5: pull_line / pull_gap / center / oppo_gap / oppo_line
6) launch_angle_bin: 발사각 구간
7) R_ideal_ft: 공기저항 없는 이상 포물선 비거리
8) attack_angle: 스윙 궤도 각도
9) swing_launch_gap = launch_angle - attack_angle
10) abs_swing_launch_gap = |swing_launch_gap|

설치
pip install pandas numpy scikit-learn xgboost matplotlib joblib

학습 + 테스트
python split_xgb_player_stand_pthrows_batted_distance.py --train savant_data_2025.csv --test savant_data_2026.csv

새 타구 1개 예측
python split_xgb_player_stand_pthrows_batted_distance.py --train savant_data_2025.csv --test savant_data_2026.csv --ev 102.4 --la 28 --spray left_gap --stand R --attack-angle 12

주의
- 실제 스핀 데이터가 아닙니다.
- swing_launch_gap은 스윙 궤도 대비 타구 발사각 차이를 나타내는 수치형 변수입니다.
- --player를 입력하면 train 데이터에서 해당 선수의 평균 attack_angle과 타석 방향을 찾아 단일 예측에 적용합니다.
- 스위치히터는 --p-throws를 입력하면 투수 손에 따른 실제 타석 방향을 train 데이터에서 추정합니다.
- 10도 <= LA < 25도: line_drive 모델
- 25도 <= LA <= 50도: fly_ball 모델
- 그 외 발사각은 학습 범위 밖입니다.
"""

from __future__ import annotations

import argparse
import json
import difflib
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise ImportError("xgboost가 설치되어 있지 않습니다. `pip install xgboost`를 먼저 실행하세요.") from exc


# -----------------------------
# Constants
# -----------------------------

TARGET = "hit_distance_sc"

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

VALID_MODEL_TYPES = ["line_drive", "fly_ball"]


# -----------------------------
# Basic utilities
# -----------------------------


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def calc_rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "n": int(len(y_true)),
        "RMSE_ft": calc_rmse(y_true, y_pred),
        "MAE_ft": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else None,
        "mean_residual_ft": float(np.mean(y_true - y_pred)),
        "median_abs_error_ft": float(np.median(np.abs(y_true - y_pred))),
        "p90_abs_error_ft": float(np.quantile(np.abs(y_true - y_pred), 0.90)),
        "pct_abs_error_gt_20ft": float(np.mean(np.abs(y_true - y_pred) > 20) * 100),
        "pct_abs_error_gt_30ft": float(np.mean(np.abs(y_true - y_pred) > 30) * 100),
    }


def normalize_distance_column(df: pd.DataFrame) -> pd.DataFrame:
    """Statcast 파일마다 비거리 컬럼명이 다를 수 있으므로 hit_distance_sc로 통일합니다."""
    df = df.copy()
    if "hit_distance_sc" in df.columns:
        return df
    if "hit_distance" in df.columns:
        df["hit_distance_sc"] = df["hit_distance"]
        return df
    if "bbdist" in df.columns:
        df["hit_distance_sc"] = df["bbdist"]
        return df
    raise ValueError("비거리 컬럼이 없습니다. hit_distance_sc, hit_distance, bbdist 중 하나가 필요합니다.")


def canonical_player_key(name: str) -> str:
    """
    Statcast player_name은 'Judge, Aaron' 형태일 수 있고,
    사용자 입력은 'Aaron Judge' 형태일 수 있어서 비교용 key를 통일합니다.
    """
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    s = str(name).strip()
    if not s:
        return ""

    # 'Last, First' -> 'First Last'
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


def _mode_or_none(values: pd.Series) -> Optional[str]:
    vals = values.dropna().astype(str)
    if vals.empty:
        return None
    return str(vals.value_counts().idxmax())


def build_player_profiles(train_df: pd.DataFrame, shrinkage_k: float = 50.0) -> Tuple[pd.DataFrame, pd.DataFrame, float, dict]:
    """
    train 데이터만 사용해서 선수별 타석 방향과 attack_angle profile을 만듭니다.
    - player_profiles: 선수별 전체 평균 attack_angle, stand 분포, 투수 손별 타석 방향
    - player_stand_profiles: 선수+타석 방향별 attack_angle 평균
    """
    if "player_name" not in train_df.columns:
        raise ValueError("--player 자동 lookup을 쓰려면 train CSV에 player_name 컬럼이 필요합니다.")

    temp = train_df.dropna(subset=["player_name", "attack_angle", "stand"]).copy()
    temp["player_key"] = temp["player_name"].apply(canonical_player_key)
    temp = temp[temp["player_key"] != ""].copy()
    temp["stand"] = temp["stand"].apply(normalize_stand)
    if "p_throws" in temp.columns:
        temp["p_throws"] = temp["p_throws"].apply(normalize_pitcher_hand)
    else:
        temp["p_throws"] = "unknown"

    if temp.empty:
        raise ValueError("선수 profile을 만들 수 없습니다. player_name/attack_angle/stand 값을 확인하세요.")

    league_mean = float(temp["attack_angle"].mean())
    league_by_stand = temp.groupby("stand")["attack_angle"].mean().to_dict()
    for st in ["R", "L"]:
        league_by_stand.setdefault(st, league_mean)

    # 선수 전체 attack angle profile
    prof = (
        temp.groupby("player_key", as_index=False)
        .agg(
            player_name_sample=("player_name", "first"),
            player_attack_angle_mean=("attack_angle", "mean"),
            player_attack_angle_median=("attack_angle", "median"),
            player_attack_angle_std=("attack_angle", "std"),
            player_attack_angle_n=("attack_angle", "size"),
        )
    )
    k = float(shrinkage_k)
    n = prof["player_attack_angle_n"].astype(float)
    prof["player_attack_angle_shrunk"] = (
        (n / (n + k)) * prof["player_attack_angle_mean"]
        + (k / (n + k)) * league_mean
    )

    # 선수별 타석 방향 분포
    stand_counts = temp.pivot_table(index="player_key", columns="stand", values="attack_angle", aggfunc="size", fill_value=0)
    for st in ["R", "L"]:
        if st not in stand_counts.columns:
            stand_counts[st] = 0
    stand_counts = stand_counts[["R", "L"]].reset_index().rename(columns={"R": "stand_R_n", "L": "stand_L_n"})
    prof = prof.merge(stand_counts, on="player_key", how="left")
    prof[["stand_R_n", "stand_L_n"]] = prof[["stand_R_n", "stand_L_n"]].fillna(0).astype(int)
    prof["primary_stand"] = np.where(prof["stand_R_n"] >= prof["stand_L_n"], "R", "L")
    prof["is_switch_profile"] = (prof["stand_R_n"] >= 5) & (prof["stand_L_n"] >= 5)

    # 선수+투수 손별 실제 타석 방향 mode
    stand_vs = (
        temp[temp["p_throws"].isin(["R", "L"])]
        .groupby(["player_key", "p_throws"])
        .agg(
            resolved_stand=("stand", _mode_or_none),
            matchup_n=("stand", "size"),
        )
        .reset_index()
    )
    if not stand_vs.empty:
        piv_stand = stand_vs.pivot(index="player_key", columns="p_throws", values="resolved_stand").reset_index()
        piv_count = stand_vs.pivot(index="player_key", columns="p_throws", values="matchup_n").reset_index()
        rename_s = {"R": "stand_vs_RHP", "L": "stand_vs_LHP"}
        rename_n = {"R": "stand_vs_RHP_n", "L": "stand_vs_LHP_n"}
        piv_stand = piv_stand.rename(columns=rename_s)
        piv_count = piv_count.rename(columns=rename_n)
        prof = prof.merge(piv_stand, on="player_key", how="left").merge(piv_count, on="player_key", how="left")
    else:
        prof["stand_vs_RHP"] = np.nan
        prof["stand_vs_LHP"] = np.nan
        prof["stand_vs_RHP_n"] = 0
        prof["stand_vs_LHP_n"] = 0

    for c in ["stand_vs_RHP_n", "stand_vs_LHP_n"]:
        if c not in prof.columns:
            prof[c] = 0
        prof[c] = prof[c].fillna(0).astype(int)

    # 선수+타석 방향별 attack angle profile
    ps = (
        temp.groupby(["player_key", "stand"], as_index=False)
        .agg(
            player_name_sample=("player_name", "first"),
            player_stand_attack_angle_mean=("attack_angle", "mean"),
            player_stand_attack_angle_median=("attack_angle", "median"),
            player_stand_attack_angle_n=("attack_angle", "size"),
        )
    )
    ps["league_stand_attack_angle_mean"] = ps["stand"].map(league_by_stand).astype(float)
    n2 = ps["player_stand_attack_angle_n"].astype(float)
    ps["player_stand_attack_angle_shrunk"] = (
        (n2 / (n2 + k)) * ps["player_stand_attack_angle_mean"]
        + (k / (n2 + k)) * ps["league_stand_attack_angle_mean"]
    )

    prof = prof.sort_values("player_attack_angle_n", ascending=False).reset_index(drop=True)
    ps = ps.sort_values(["player_key", "stand"]).reset_index(drop=True)
    return prof, ps, league_mean, {str(k): float(v) for k, v in league_by_stand.items()}


def lookup_player_row(player_profiles: pd.DataFrame, player_name: str) -> Optional[pd.Series]:
    if player_profiles is None or player_profiles.empty:
        return None
    key = canonical_player_key(player_name)
    if not key:
        return None
    keys = player_profiles["player_key"].astype(str).tolist()
    match_key = None
    if key in set(keys):
        match_key = key
    else:
        close = difflib.get_close_matches(key, keys, n=1, cutoff=0.78)
        if close:
            match_key = close[0]
    if match_key is None:
        return None
    return player_profiles[player_profiles["player_key"] == match_key].iloc[0]


def resolve_stand_for_single(
    stand: Optional[str],
    player: Optional[str],
    p_throws: Optional[str],
    player_profiles: pd.DataFrame,
) -> Tuple[str, str, str, int, bool]:
    """
    단일 예측에서 타석 방향을 결정합니다.
    반환: stand, source, matched_name, matched_n, is_switch_profile
    """
    if stand is not None:
        return normalize_stand(stand), "manual_input", str(player or "manual_input"), 0, False

    if player is None:
        raise ValueError("--stand를 직접 입력하거나 --player를 입력해야 합니다.")

    row = lookup_player_row(player_profiles, player)
    if row is None:
        raise ValueError(f"선수 '{player}'를 train 데이터에서 찾지 못했습니다. --stand를 직접 입력하세요.")

    matched_name = str(row["player_name_sample"])
    n = int(row["player_attack_angle_n"])
    is_switch = bool(row.get("is_switch_profile", False))

    if is_switch:
        if p_throws is None:
            raise ValueError(
                f"{matched_name}은 train 데이터 기준 스위치히터로 보입니다. "
                "--p-throws R 또는 --p-throws L을 입력하거나 --stand를 직접 입력하세요."
            )
        ph = normalize_pitcher_hand(p_throws)
        col = "stand_vs_RHP" if ph == "R" else "stand_vs_LHP" if ph == "L" else None
        if col and pd.notna(row.get(col, np.nan)):
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
    """
    사용자 입력 선수명과 타석 방향으로 train 기반 평균 attack_angle을 찾습니다.
    반환: attack_angle, matched_name, n, source
    """
    row = lookup_player_row(player_profiles, player_name)
    if row is None:
        return float(league_attack_angle), "league_average", 0, "league_fallback_not_found"

    player_key = str(row["player_key"])
    matched_name = str(row["player_name_sample"])
    st = normalize_stand(stand)

    ps = player_stand_profiles[
        (player_stand_profiles["player_key"] == player_key)
        & (player_stand_profiles["stand"] == st)
    ]
    if not ps.empty:
        r = ps.iloc[0]
        n = int(r["player_stand_attack_angle_n"])
        source = "player_stand_shrunk" if n >= int(min_player_bbe) else "player_stand_shrunk_low_n"
        return float(r["player_stand_attack_angle_shrunk"]), matched_name, n, source

    # 해당 타석 방향 표본이 없으면 선수 전체 shrunk 평균 사용
    n = int(row["player_attack_angle_n"])
    source = "player_overall_shrunk_no_stand_profile" if n >= int(min_player_bbe) else "player_overall_shrunk_low_n"
    return float(row["player_attack_angle_shrunk"]), matched_name, n, source


# -----------------------------
# Feature engineering
# -----------------------------


def calc_spray_angle_from_hc(df: pd.DataFrame) -> pd.Series:
    """
    Statcast hc_x, hc_y로부터 spray angle을 근사합니다.
    음수: 좌측, 0 근처: 중앙, 양수: 우측
    """
    return np.degrees(
        np.arctan2(
            df["hc_x"].astype(float) - 125.42,
            198.27 - df["hc_y"].astype(float),
        )
    )


def angle_to_spray_zone5(angle: float) -> str:
    """연속 spray angle을 5개 구간으로 변환합니다."""
    if pd.isna(angle):
        return "unknown"
    a = float(angle)
    if a < -27.5:
        return "left_line"
    if a < -7.5:
        return "left_gap"
    if a <= 7.5:
        return "center"
    if a <= 27.5:
        return "right_gap"
    return "right_line"


def normalize_spray_zone5(zone: str) -> str:
    """사용자 입력을 5개 spray zone 표준값으로 변환합니다."""
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

    valid = ", ".join(SPRAY_REP_ANGLE.keys())
    raise ValueError(f"spray는 다음 중 하나여야 합니다: {valid}")


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
    raise ValueError("p_throws는 R 또는 L이어야 합니다.")


def infer_matchup_side(stand: str, p_throws: str) -> str:
    st = normalize_stand(stand)
    ph = normalize_pitcher_hand(p_throws)
    if ph not in ["R", "L"]:
        return "unknown"
    return "same_hand" if st == ph else "opposite_hand"


def classify_batted_direction5(spray_zone5: str, stand: str) -> str:
    """절대 좌/우 방향을 타자 기준 pull/oppo 방향으로 변환합니다."""
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
    """
    공기저항 없는 이상 포물선 비거리입니다.
    실제 비거리가 아니라, EV와 LA를 물리적으로 조합한 보조 feature입니다.
    """
    g = 9.81
    v0 = np.asarray(launch_speed_mph, dtype=float) * 0.44704
    theta = np.radians(np.asarray(launch_angle_deg, dtype=float))

    vx = v0 * np.cos(theta)
    vz = v0 * np.sin(theta)

    t_flight = (vz + np.sqrt(np.maximum(vz ** 2 + 2 * g * z0_m, 0))) / g
    r_m = vx * t_flight
    return r_m * 3.28084


def add_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """최종 모델에 들어갈 feature를 생성합니다."""
    df = df.copy()

    df["stand"] = df["stand"].apply(normalize_stand)
    if "p_throws" not in df.columns:
        df["p_throws"] = "unknown"
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

    # 실제 스핀 변수는 아니며, 스윙 궤도 대비 타구 발사각 차이만 수치형으로 사용합니다.
    df["attack_angle"] = pd.to_numeric(df["attack_angle"], errors="coerce")
    df["swing_launch_gap"] = df["launch_angle"].astype(float) - df["attack_angle"].astype(float)
    df["abs_swing_launch_gap"] = df["swing_launch_gap"].abs()

    return df


def clean_training_data(raw_df: pd.DataFrame, exclude_parks: Iterable[str], max_abs_spray: float) -> pd.DataFrame:
    """
    학습/테스트 데이터 정리.
    원본 Statcast의 hc_x, hc_y는 spray_zone5 생성에만 쓰고,
    최종 예측 입력에서는 사용하지 않습니다.
    """
    df = normalize_distance_column(raw_df)
    df = df.copy()

    required = ["launch_speed", "launch_angle", "hit_distance_sc", "hc_x", "hc_y", "stand", "attack_angle"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")
    if "p_throws" not in df.columns:
        df["p_throws"] = "unknown"

    exclude_parks = list(exclude_parks or [])
    if exclude_parks and "home_team" in df.columns:
        df = df[~df["home_team"].isin(exclude_parks)].copy()

    df = df.dropna(subset=required).copy()

    df = df[(df["launch_speed"] >= 30) & (df["launch_speed"] <= 125)].copy()
    df = df[(df["launch_angle"] >= 10) & (df["launch_angle"] <= 50)].copy()
    df = df[(df["hit_distance_sc"] > 0) & (df["hit_distance_sc"] <= 500)].copy()

    # 원본 bb_type이 있으면 line_drive/fly_ball만 유지합니다.
    # 단, 최종 모델 타입은 사용자 입력과 맞추기 위해 launch_angle로 다시 판단합니다.
    if "bb_type" in df.columns:
        df = df[df["bb_type"].isin(["fly_ball", "line_drive"])].copy()

    df["spray_angle_raw"] = calc_spray_angle_from_hc(df)
    df = df[df["spray_angle_raw"].between(-float(max_abs_spray), float(max_abs_spray))].copy()
    df["spray_zone5"] = df["spray_angle_raw"].apply(angle_to_spray_zone5)
    df = df[df["spray_zone5"].isin(SPRAY_REP_ANGLE.keys())].copy()

    df = add_model_features(df)
    df = df[df["model_type"].isin(VALID_MODEL_TYPES)].copy()

    return df.reset_index(drop=True)


# -----------------------------
# XGBoost matrix and model
# -----------------------------


def prepare_xgb_matrix(train_df: pd.DataFrame, pred_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], pd.Series]:
    """한 모델 타입(line/fly)에 대한 XGBoost 입력 행렬을 생성합니다."""
    x_train_num = train_df[NUMERIC_FEATURES].copy()
    x_pred_num = pred_df[NUMERIC_FEATURES].copy()

    medians = x_train_num.median(numeric_only=True)
    x_train_num = x_train_num.fillna(medians)
    x_pred_num = x_pred_num.fillna(medians)

    x_train_cat = train_df[CATEGORICAL_FEATURES].copy().fillna("missing").astype(str)
    x_pred_cat = pred_df[CATEGORICAL_FEATURES].copy().fillna("missing").astype(str)

    x_train_cat = pd.get_dummies(x_train_cat, columns=CATEGORICAL_FEATURES)
    x_pred_cat = pd.get_dummies(x_pred_cat, columns=CATEGORICAL_FEATURES)
    x_pred_cat = x_pred_cat.reindex(columns=x_train_cat.columns, fill_value=0)

    x_train = pd.concat([x_train_num.reset_index(drop=True), x_train_cat.reset_index(drop=True)], axis=1)
    x_pred = pd.concat([x_pred_num.reset_index(drop=True), x_pred_cat.reset_index(drop=True)], axis=1)

    feature_columns = list(x_train.columns)
    return x_train, x_pred, feature_columns, medians


def prepare_prediction_matrix(df: pd.DataFrame, feature_columns: List[str], medians: pd.Series) -> pd.DataFrame:
    x_num = df[NUMERIC_FEATURES].copy().fillna(medians)
    x_cat = df[CATEGORICAL_FEATURES].copy().fillna("missing").astype(str)
    x_cat = pd.get_dummies(x_cat, columns=CATEGORICAL_FEATURES)
    x = pd.concat([x_num.reset_index(drop=True), x_cat.reset_index(drop=True)], axis=1)
    x = x.reindex(columns=feature_columns, fill_value=0)
    return x


def make_xgb_model(args) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        min_child_weight=args.min_child_weight,
        gamma=args.gamma,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )


def fit_split_models(train: pd.DataFrame, test: pd.DataFrame, args) -> Tuple[Dict[str, dict], pd.Series]:
    """line_drive 모델과 fly_ball 모델을 따로 학습하고 test를 자동 라우팅해 예측합니다."""
    model_bundles: Dict[str, dict] = {}
    pred_all = pd.Series(index=test.index, dtype=float)

    for model_type in VALID_MODEL_TYPES:
        tr = train[train["model_type"] == model_type].copy()
        te = test[test["model_type"] == model_type].copy()

        if len(tr) < args.min_rows_per_model:
            raise ValueError(f"{model_type} 학습 데이터가 너무 적습니다: {len(tr)} rows")
        if len(te) == 0:
            print(f"[INFO] test에 {model_type} 행이 없습니다. 모델은 학습만 합니다.")

        x_train, x_test, feature_columns, medians = prepare_xgb_matrix(tr, te if len(te) else tr.head(1))
        y_train = tr[TARGET]

        model = make_xgb_model(args)
        model.fit(x_train, y_train, verbose=False)

        if len(te):
            x_test = prepare_prediction_matrix(te, feature_columns, medians)
            pred_all.loc[te.index] = model.predict(x_test)

        model_bundles[model_type] = {
            "model": model,
            "feature_columns": feature_columns,
            "medians": medians,
            "train_rows": int(len(tr)),
            "test_rows": int(len(te)),
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
        }

    if pred_all.isna().any():
        missing = int(pred_all.isna().sum())
        raise ValueError(f"예측되지 않은 test 행이 있습니다: {missing}")

    return model_bundles, pred_all


# -----------------------------
# Single input prediction
# -----------------------------


def make_single_input(ev: float, la: float, spray: str, stand: str, p_throws: Optional[str], attack_angle: float) -> pd.DataFrame:
    row = pd.DataFrame(
        [
            {
                "launch_speed": float(ev),
                "launch_angle": float(la),
                "spray_zone5": normalize_spray_zone5(spray),
                "stand": normalize_stand(stand),
                "p_throws": normalize_pitcher_hand(p_throws) if p_throws is not None else "unknown",
                "attack_angle": float(attack_angle),
            }
        ]
    )
    return add_model_features(row)


def resolve_attack_angle_for_single(
    attack_angle: Optional[float],
    player: Optional[str],
    stand: str,
    player_profiles: pd.DataFrame,
    player_stand_profiles: pd.DataFrame,
    league_attack_angle: float,
    min_player_bbe: int,
) -> Tuple[float, str, str, int]:
    """단일 예측에서 attack_angle을 직접 입력값 또는 선수+타석 방향 평균값으로 결정합니다."""
    if attack_angle is not None:
        return float(attack_angle), "manual_input", str(player or "manual_input"), 0

    if player is not None:
        aa, matched_name, n, source = lookup_player_attack_angle(
            player_profiles,
            player_stand_profiles,
            player,
            stand,
            league_attack_angle,
            min_player_bbe=min_player_bbe,
        )
        return aa, source, matched_name, n

    raise ValueError("단일 예측에는 --attack-angle 또는 --player 중 하나가 필요합니다.")


def predict_single(
    model_bundles: Dict[str, dict],
    ev: float,
    la: float,
    spray: str,
    stand: Optional[str],
    p_throws: Optional[str],
    attack_angle: Optional[float],
    player: Optional[str],
    player_profiles: pd.DataFrame,
    player_stand_profiles: pd.DataFrame,
    league_attack_angle: float,
    min_player_bbe: int,
) -> Tuple[pd.DataFrame, float, dict]:
    resolved_stand, stand_source, matched_stand_name, stand_n, is_switch = resolve_stand_for_single(
        stand, player, p_throws, player_profiles
    )
    resolved_attack_angle, aa_source, matched_aa_name, aa_n = resolve_attack_angle_for_single(
        attack_angle, player, resolved_stand, player_profiles, player_stand_profiles, league_attack_angle, min_player_bbe
    )

    single = make_single_input(ev, la, spray, resolved_stand, p_throws, resolved_attack_angle)
    single["stand_source"] = stand_source
    single["attack_angle_source"] = aa_source
    single["matched_player_name"] = matched_aa_name if matched_aa_name != "league_average" else matched_stand_name
    single["matched_player_attack_angle_n"] = aa_n
    single["matched_player_stand_n"] = stand_n
    single["is_switch_profile"] = is_switch

    model_type = single.loc[0, "model_type"]
    if model_type not in model_bundles:
        raise ValueError("이 발사각은 학습 범위 밖입니다. 10~50도 범위를 권장합니다.")

    bundle = model_bundles[model_type]
    x_single = prepare_prediction_matrix(single, bundle["feature_columns"], bundle["medians"])
    pred = float(bundle["model"].predict(x_single)[0])
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


# -----------------------------
# Output helpers
# -----------------------------


def save_actual_vs_predicted(y_true, y_pred, outpath: Path, title: str) -> None:
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, alpha=0.25)
    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Actual distance (ft)")
    plt.ylabel("Predicted distance (ft)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def summarize_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    return (
        df.groupby(group_col, observed=False)
        .agg(
            n=("residual_ft", "size"),
            mean_residual_ft=("residual_ft", "mean"),
            MAE_ft=("abs_error_ft", "mean"),
            RMSE_ft=("residual_ft", lambda x: np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2))),
            p90_abs_error_ft=("abs_error_ft", lambda x: np.quantile(np.asarray(x, dtype=float), 0.90)),
        )
        .reset_index()
    )


def save_summaries(test_out: pd.DataFrame, outdir: Path) -> None:
    for col in ["model_type", "launch_angle_bin", "spray_zone5", "batted_direction5", "stand"]:
        summarize_group(test_out, col).to_csv(outdir / f"residual_summary_by_{col}.csv", index=False, encoding="utf-8-sig")

    ev_bins = [0, 70, 80, 90, 100, 105, 110, 115, 130]
    temp = test_out.copy()
    temp["EV_bin"] = pd.cut(temp["launch_speed"], bins=ev_bins, include_lowest=True)
    summarize_group(temp, "EV_bin").to_csv(outdir / "residual_summary_by_EV_bin.csv", index=False, encoding="utf-8-sig")


def save_high_error_audit(test_out: pd.DataFrame, outdir: Path) -> None:
    """큰 오차가 난 타구를 복기할 수 있도록 별도 CSV로 저장합니다."""
    review_cols = [
        "game_date", "player_name", "events", "des",
        "home_team", "away_team", "stand", "bb_type", "model_type",
        "launch_speed", "launch_angle", "attack_angle", "swing_launch_gap", "abs_swing_launch_gap",
        "spray_zone5", "batted_direction5", "launch_angle_bin",
        "hit_location", TARGET, "pred_distance_ft", "residual_ft", "abs_error_ft",
        "hc_x", "hc_y", "estimated_ba_using_speedangle", "launch_speed_angle",
    ]
    review_cols = [c for c in review_cols if c in test_out.columns]

    high_error_40 = test_out[test_out["abs_error_ft"] >= 40].sort_values("abs_error_ft", ascending=False)
    high_error_60 = test_out[test_out["abs_error_ft"] >= 60].sort_values("abs_error_ft", ascending=False)
    line_drive_high_error = test_out[
        (test_out["model_type"] == "line_drive") & (test_out["abs_error_ft"] >= 40)
    ].sort_values("abs_error_ft", ascending=False)

    high_error_40[review_cols].to_csv(outdir / "high_error_over_40ft_review.csv", index=False, encoding="utf-8-sig")
    high_error_60[review_cols].to_csv(outdir / "high_error_over_60ft_review.csv", index=False, encoding="utf-8-sig")
    line_drive_high_error[review_cols].to_csv(outdir / "line_drive_high_error_review.csv", index=False, encoding="utf-8-sig")

    rows = []
    for name, data in {
        "all_over_40ft": high_error_40,
        "all_over_60ft": high_error_60,
        "line_drive_over_40ft": line_drive_high_error,
    }.items():
        rows.append({
            "group": name,
            "n": int(len(data)),
            "mean_abs_error_ft": float(data["abs_error_ft"].mean()) if len(data) else None,
            "mean_residual_ft": float(data["residual_ft"].mean()) if len(data) else None,
            "mean_launch_speed": float(data["launch_speed"].mean()) if len(data) else None,
            "mean_launch_angle": float(data["launch_angle"].mean()) if len(data) else None,
            "mean_attack_angle": float(data["attack_angle"].mean()) if "attack_angle" in data.columns and len(data) else None,
            "mean_swing_launch_gap": float(data["swing_launch_gap"].mean()) if "swing_launch_gap" in data.columns and len(data) else None,
            "mean_actual_distance": float(data[TARGET].mean()) if len(data) else None,
            "mean_pred_distance": float(data["pred_distance_ft"].mean()) if len(data) else None,
        })

    pd.DataFrame(rows).to_csv(outdir / "high_error_audit_summary.csv", index=False, encoding="utf-8-sig")


def json_safe(obj):
    """json.dump가 pandas/numpy 객체를 처리하도록 변환합니다."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# -----------------------------
# Main
# -----------------------------


def run(args) -> None:
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    train_raw = read_csv(args.train)
    test_raw = read_csv(args.test)

    train = clean_training_data(
        train_raw,
        exclude_parks=args.exclude_parks,
        max_abs_spray=args.max_abs_spray,
    )
    test = clean_training_data(
        test_raw,
        exclude_parks=args.exclude_parks,
        max_abs_spray=args.max_abs_spray,
    )

    player_profiles, player_stand_profiles, league_attack_angle, league_attack_angle_by_stand = build_player_profiles(
        train, shrinkage_k=args.player_profile_shrinkage
    )

    model_bundles, pred = fit_split_models(train, test, args)

    test_out = test.copy()
    test_out["pred_distance_ft"] = pred.values
    test_out["residual_ft"] = test_out[TARGET] - test_out["pred_distance_ft"]
    test_out["abs_error_ft"] = test_out["residual_ft"].abs()

    overall_metrics = evaluate(test_out[TARGET], test_out["pred_distance_ft"])
    by_type_metrics = {
        mt: evaluate(g[TARGET], g["pred_distance_ft"])
        for mt, g in test_out.groupby("model_type")
    }

    metrics = {
        "overall": overall_metrics,
        "by_model_type": by_type_metrics,
        "train_rows_total": int(len(train)),
        "test_rows_total": int(len(test)),
        "excluded_parks": list(args.exclude_parks),
        "max_abs_spray": float(args.max_abs_spray),
        "model_train_rows": {mt: bundle["train_rows"] for mt, bundle in model_bundles.items()},
        "model_test_rows": {mt: bundle["test_rows"] for mt, bundle in model_bundles.items()},
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "league_attack_angle_mean": float(league_attack_angle),
        "league_attack_angle_by_stand": league_attack_angle_by_stand,
        "player_profile_shrinkage": float(args.player_profile_shrinkage),
        "player_profiles_n": int(len(player_profiles)),
        "player_stand_profiles_n": int(len(player_stand_profiles)),
    }

    # 저장용 bundle에서 pandas Series는 joblib로는 문제없지만, json에는 별도 저장합니다.
    joblib.dump(
        {
            "models": model_bundles,
            "spray_rep_angle": SPRAY_REP_ANGLE,
            "valid_model_types": VALID_MODEL_TYPES,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "target": TARGET,
            "routing_rule": "10 <= LA < 25: line_drive; 25 <= LA <= 50: fly_ball",
            "player_profiles": player_profiles,
            "player_stand_profiles": player_stand_profiles,
            "league_attack_angle_mean": league_attack_angle,
            "league_attack_angle_by_stand": league_attack_angle_by_stand,
            "player_profile_shrinkage": args.player_profile_shrinkage,
        },
        outdir / "split_xgb_batted_distance.joblib",
    )

    train.to_csv(outdir / "cleaned_train_split_xgb.csv", index=False, encoding="utf-8-sig")
    test_out.to_csv(outdir / "test_predictions_split_xgb.csv", index=False, encoding="utf-8-sig")
    player_profiles.to_csv(outdir / "player_profiles_train_only.csv", index=False, encoding="utf-8-sig")
    player_stand_profiles.to_csv(outdir / "player_stand_attack_angle_profiles_train_only.csv", index=False, encoding="utf-8-sig")

    with open(outdir / "metrics_split_xgb.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=json_safe)

    feature_cols = {mt: bundle["feature_columns"] for mt, bundle in model_bundles.items()}
    with open(outdir / "feature_columns_split_xgb.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)

    save_actual_vs_predicted(
        test_out[TARGET],
        test_out["pred_distance_ft"],
        outdir / "actual_vs_predicted_split_xgb.png",
        "Actual vs predicted: split XGBoost",
    )
    save_summaries(test_out, outdir)
    save_high_error_audit(test_out, outdir)

    print()
    print("=== Split XGBoost batted distance model ===")
    print(f"Excluded parks: {list(args.exclude_parks)}")
    print(f"Train rows: {len(train):,}")
    print(f"Test rows:  {len(test):,}")
    print("Model routing:")
    print("  10 <= LA < 25  -> line_drive model")
    print("  25 <= LA <= 50 -> fly_ball model")
    print("Features:")
    print("  numeric:", NUMERIC_FEATURES)
    print("  categorical:", CATEGORICAL_FEATURES)
    print(f"Player profiles: {len(player_profiles):,} players")
    print(f"Player+stand attack angle profiles: {len(player_stand_profiles):,} rows")
    print(f"League mean attack angle: {league_attack_angle:.2f} deg")

    print()
    print("=== Overall test metrics ===")
    for k, v in overall_metrics.items():
        print(f"{k}: {v}")

    print()
    print("=== Metrics by model type ===")
    for mt, m in by_type_metrics.items():
        print(f"[{mt}]")
        for k, v in m.items():
            print(f"  {k}: {v}")

    if (
        args.ev is not None
        and args.la is not None
        and args.spray is not None
        and (args.stand is not None or args.player is not None)
        and (args.attack_angle is not None or args.player is not None)
    ):
        single, pred_single, single_info = predict_single(
            model_bundles,
            args.ev,
            args.la,
            args.spray,
            args.stand,
            args.p_throws,
            args.attack_angle,
            args.player,
            player_profiles,
            player_stand_profiles,
            league_attack_angle,
            args.min_player_bbe,
        )

        print()
        print("=== Single batted-ball prediction ===")
        print(f"Input EV: {args.ev} mph")
        print(f"Input LA: {args.la} deg")
        print(f"Input spray_zone5: {single.loc[0, 'spray_zone5']}")
        print(f"Input/resolved stand: {single.loc[0, 'stand']}")
        print(f"Pitcher throws: {single.loc[0, 'p_throws']}")
        print(f"Matchup side: {single.loc[0, 'matchup_side']}")
        if args.player is not None:
            print(f"Input player: {args.player}")
            print(f"Matched player: {single_info['matched_player_name']}")
            print(f"Stand source: {single_info['stand_source']}")
            print(f"Player stand sample n: {single_info['matched_player_stand_n']}")
            print(f"Switch-hitter profile: {single_info['is_switch_profile']}")
            print(f"Player attack angle sample n: {single_info['matched_player_attack_angle_n']}")
            print(f"Attack angle source: {single_info['attack_angle_source']}")
        print(f"Applied attack_angle: {single.loc[0, 'attack_angle']:.2f} deg")
        print(f"Selected model: {single.loc[0, 'model_type']}")
        print(f"LA bin: {single.loc[0, 'launch_angle_bin']}")
        print(f"Batted direction: {single.loc[0, 'batted_direction5']}")
        print(f"Swing-launch gap (LA - attack_angle): {single.loc[0, 'swing_launch_gap']:.2f} deg")
        print(f"R_ideal_ft: {single.loc[0, 'R_ideal_ft']:.2f} ft")
        print(f"Predicted distance: {pred_single:.1f} ft")

    print()
    print("Outputs saved to:")
    print(outdir.resolve())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Training CSV, e.g. savant_data_2025.csv")
    parser.add_argument("--test", required=True, help="Test CSV, e.g. savant_data_2026.csv")
    parser.add_argument("--output", default="split_xgb_player_stand_pthrows_results")
    parser.add_argument(
        "--exclude-parks",
        nargs="*",
        default=["COL", "SF"],
        help="제외할 home_team 코드. 기본값: COL SF. 제외하지 않으려면 --exclude-parks 만 입력하세요.",
    )
    parser.add_argument("--max-abs-spray", type=float, default=45.0)
    parser.add_argument("--min-rows-per-model", type=int, default=300)
    parser.add_argument("--player-profile-shrinkage", type=float, default=50.0, help="선수 평균 attack_angle을 리그 평균 쪽으로 당기는 shrinkage 강도")
    parser.add_argument("--min-player-bbe", type=int, default=5, help="선수 lookup 시 표본 수가 이보다 작으면 shrunk 값으로 표시합니다.")

    # 과적합을 줄이기 위해 simple 모델보다 조금 더 보수적인 기본값을 사용합니다.
    parser.add_argument("--n-estimators", type=int, default=650)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample-bytree", type=float, default=0.85)
    parser.add_argument("--reg-lambda", type=float, default=3.0)
    parser.add_argument("--min-child-weight", type=float, default=8.0)
    parser.add_argument("--gamma", type=float, default=0.0)

    parser.add_argument("--ev", type=float, default=None, help="새 타구 예측용 타구속도, mph")
    parser.add_argument("--la", type=float, default=None, help="새 타구 예측용 발사각, deg")
    parser.add_argument(
        "--spray",
        type=str,
        default=None,
        help="새 타구 예측용 스프레이 방향: left_line/left_gap/center/right_gap/right_line",
    )
    parser.add_argument("--stand", type=str, default=None, help="새 타구 예측용 타자 방향: R/L. --player로 자동 추정 가능")
    parser.add_argument("--p-throws", type=str, default=None, help="새 타구 예측용 투수 손: R/L. 스위치히터 stand 자동 추정에 필요")
    parser.add_argument("--attack-angle", type=float, default=None, help="새 타구 예측용 attack angle, deg. 입력하면 --player보다 우선합니다.")
    parser.add_argument("--player", type=str, default=None, help="새 타구 예측용 선수명. 예: 'Aaron Judge'. attack_angle 미입력 시 train 기반 선수 평균을 적용합니다.")

    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
