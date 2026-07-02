r"""
batted_ball_simulator_app_paths_v2.py

Streamlit UI for the batted-ball distance simulator.
사용자 PC 경로에 맞춘 실행용 UI 코드입니다.

기본 경로
- 모델 결과 폴더:
  C:\Users\모규현\Desktop\batted_ball_simulator\model_results_3
- 학습 코드 파일:
  C:\Users\모규현\Desktop\batted_ball_simulator\split_xgb_batted_distance_3.py

실행
    streamlit run batted_ball_simulator_app_paths_v2.py
"""

from __future__ import annotations

import importlib.util
import re
import sys
import unicodedata
from pathlib import Path
from types import ModuleType
from typing import List, Optional

import joblib
import pandas as pd
import streamlit as st
from io import StringIO
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_MODEL_RESULTS_DIR = BASE_DIR
DEFAULT_CODE_FILE = BASE_DIR / "split_xgb_batted_distance_3.py"
DEFAULT_MODEL_FILE = BASE_DIR / "split_xgb_batted_distance.joblib"

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

PARK_OPTIONS = [
    "미적용",
    "Yankee Stadium",
    "Dodger Stadium",
    "Fenway Park",
    "Citi Field",
    "Petco Park",
    "기타",
]

# ------------------------------------------------------------
# Streamlit 기본 설정
# ------------------------------------------------------------

st.set_page_config(
    page_title="Batted Ball Distance Simulator",
    page_icon="⚾",
    layout="wide",
)

st.title("⚾ Batted Ball Distance Simulator")
st.caption("XGBoost 기반 타구 비거리 예측 UI")

# ------------------------------------------------------------
# 경로 / 모듈 로딩 함수
# ------------------------------------------------------------


def resolve_model_path(model_results_path: str) -> Path:
    """
    사용자가 폴더를 넣으면 그 안에서 joblib 파일을 찾고,
    파일을 직접 넣으면 그대로 사용합니다.
    """
    path = Path(model_results_path).expanduser()

    if path.is_file():
        return path

    if path.is_dir():
        # 1순위: 정해진 파일명
        for filename in DEFAULT_MODEL_FILENAMES:
            candidate = path / filename
            if candidate.exists():
                return candidate

        # 2순위: 폴더 안의 첫 번째 joblib
        joblib_files = sorted(path.glob("*.joblib"))
        if joblib_files:
            return joblib_files[0]

        # 3순위: 하위 폴더까지 검색
        recursive_joblib_files = sorted(path.rglob("*.joblib"))
        if recursive_joblib_files:
            return recursive_joblib_files[0]

    # 경로가 아직 없거나 파일을 못 찾은 경우, 기본 예상 경로를 반환해 에러 메시지를 명확히 함
    return path / "split_xgb_batted_distance.joblib" if not path.suffix else path



def resolve_code_file(code_path: str) -> Path:
    """
    사용자가 코드 폴더를 넣으면 split_xgb_player_stand_pthrows_batted_distance.py를 찾고,
    .py 파일을 직접 넣으면 그대로 사용합니다.
    """
    path = Path(code_path).expanduser()

    if path.is_file() and path.suffix == ".py":
        return path

    if path.is_dir():
        candidate = path / DEFAULT_CODE_FILENAME
        if candidate.exists():
            return candidate

        # 혹시 파일명이 조금 다를 경우, 관련 py 파일 탐색
        py_candidates = sorted(path.glob("*pthrows*batted_distance*.py"))
        if py_candidates:
            return py_candidates[0]

        py_candidates = sorted(path.glob("split_xgb*.py"))
        if py_candidates:
            return py_candidates[0]

    return path / DEFAULT_CODE_FILENAME if not path.suffix else path


@st.cache_resource(show_spinner=False)
def load_model_module(code_path: str) -> ModuleType:
    code_file = resolve_code_file(code_path)
    if not code_file.exists():
        raise FileNotFoundError(f"학습 코드 파일을 찾지 못했습니다: {code_file}")

    code_dir = str(code_file.parent)
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)

    spec = importlib.util.spec_from_file_location("batted_ball_model_code", code_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"학습 코드 파일을 import할 수 없습니다: {code_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["batted_ball_model_code"] = module
    spec.loader.exec_module(module)
    return module


@st.cache_resource(show_spinner=False)
def load_bundle(model_results_path: str) -> dict:
    model_path = resolve_model_path(model_results_path)
    if not model_path.exists():
        raise FileNotFoundError(f"모델 joblib 파일을 찾지 못했습니다: {model_path}")
    return joblib.load(model_path)


# ------------------------------------------------------------
# UI 보조 함수
# ------------------------------------------------------------


def display_player_name(raw_name: str) -> str:
    """Statcast의 'Last, First' 이름을 UI에서 보기 쉬운 'First Last'로 바꿉니다."""
    s = str(raw_name).strip()
    if "," in s:
        last, first = [p.strip() for p in s.split(",", 1)]
        if first and last:
            return f"{first} {last}"
    return s



def normalize_text_for_search(text: str) -> str:
    s = unicodedata.normalize("NFKD", str(text))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


@st.cache_data(show_spinner=False)
def build_player_options(player_profiles_json: str) -> pd.DataFrame:
    profiles = pd.read_json(StringIO(player_profiles_json))
    if profiles.empty:
        return pd.DataFrame(
            columns=["display_name", "player_key", "player_name_sample", "search_key", "n", "is_switch"]
        )

    out = profiles.copy()
    out["display_name"] = out["player_name_sample"].apply(display_player_name)
    out["search_key"] = out["display_name"].apply(normalize_text_for_search)
    out["n"] = out.get("player_attack_angle_n", 0).fillna(0).astype(int)
    out["is_switch"] = out.get("is_switch_profile", False).fillna(False).astype(bool)
    out = out.sort_values(["display_name", "n"], ascending=[True, False])

    return out[["display_name", "player_key", "player_name_sample", "search_key", "n", "is_switch"]].reset_index(drop=True)



def filter_player_candidates(options: pd.DataFrame, query: str, canonical_player_key, limit: int = 25) -> List[str]:
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



def get_profile_summary(player_profiles: pd.DataFrame, selected_raw_name: str, canonical_player_key) -> Optional[pd.Series]:
    key = canonical_player_key(selected_raw_name)
    if not key:
        return None
    rows = player_profiles[player_profiles["player_key"].astype(str) == key]
    if rows.empty:
        return None
    return rows.iloc[0]



def make_warning(ev: float, la: float, model_type: str) -> Optional[str]:
    if model_type == "line_drive" and ev >= 100 and 10 <= la <= 20:
        return "고속·저발사각 라인드라이브 구간입니다. 실제 스핀, 수비 개입, projected distance 영향으로 오차가 커질 수 있습니다."
    if la < 10 or la > 50:
        return "현재 모델의 권장 발사각 범위는 10~50도입니다."
    return None


# ------------------------------------------------------------
# Sidebar: 경로 입력 및 로딩
# ------------------------------------------------------------

with st.sidebar:
    st.header("경로 설정")

    code_path_input = st.text_input(
        "XGBoost 학습 코드 .py 파일 경로",
        value=str(DEFAULT_CODE_FILE),
        help="예: C:\\Users\\모규현\\Desktop\\batted_ball_simulator\\split_xgb_batted_distance_3.py",
    )

    model_results_input = st.text_input(
        "모델 결과 폴더 또는 .joblib 파일 경로",
        value=str(DEFAULT_MODEL_RESULTS_DIR),
        help="split_xgb_batted_distance.joblib가 들어있는 폴더입니다.",
    )

    st.caption("경로가 맞으면 앱이 자동으로 코드와 모델을 불러옵니다.")

try:
    model_code = load_model_module(code_path_input)
except Exception as exc:
    st.error("XGBoost 학습 코드 파일을 불러오지 못했습니다.")
    st.code(str(exc))
    st.info(
        "입력한 코드 경로가 실제 .py 파일을 가리키는지 확인하세요."
    )
    st.stop()

try:
    bundle = load_bundle(model_results_input)
except Exception as exc:
    st.error("모델 joblib 파일을 불러오지 못했습니다.")
    st.code(str(exc))
    st.info(
        "입력한 모델 결과 경로에 `split_xgb_batted_distance.joblib` 파일이 있는지 확인하세요."
    )
    st.stop()

try:
    canonical_player_key = model_code.canonical_player_key
    predict_single = model_code.predict_single
except AttributeError as exc:
    st.error("학습 코드 안에 필요한 함수가 없습니다.")
    st.code(str(exc))
    st.info("최신 버전의 `split_xgb_player_stand_pthrows_batted_distance.py` 파일인지 확인하세요.")
    st.stop()

try:
    models = bundle["models"]
    player_profiles = bundle["player_profiles"]
    player_stand_profiles = bundle["player_stand_profiles"]
    league_attack_angle = float(bundle["league_attack_angle_mean"])
except KeyError as exc:
    st.error("모델 joblib 파일 구조가 현재 UI 코드와 맞지 않습니다.")
    st.code(f"누락된 key: {exc}")
    st.info("최신 학습 코드로 모델 joblib를 다시 생성했는지 확인하세요.")
    st.stop()

player_options = build_player_options(player_profiles.to_json(orient="records", force_ascii=False))

with st.sidebar:
    st.success("코드/모델 로딩 완료")
    st.caption(f"사용 코드: {resolve_code_file(code_path_input)}")
    st.caption(f"사용 모델: {resolve_model_path(model_results_input)}")

# ------------------------------------------------------------
# Main UI
# ------------------------------------------------------------

left, right = st.columns([1.1, 0.9])

with left:
    st.subheader("입력")

    player_query = st.text_input(
        "1. 선수 이름 검색",
        value="",
        placeholder="예: Aaron Judge",
        help="입력하면 아래 후보 목록이 줄어듭니다.",
    )

    candidates = filter_player_candidates(player_options, player_query, canonical_player_key, limit=30)
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

    with st.expander("6. 어디서 쳤는지 / 구장 입력 예정"):
        park = st.selectbox(
            "구장",
            options=PARK_OPTIONS,
            index=0,
            help="현재 모델에는 아직 반영하지 않습니다. 추후 park factor나 담장 거리 모델을 붙일 자리입니다.",
        )
        st.caption("현재 버전에서는 예측에 사용하지 않습니다.")

    predict_clicked = st.button("예측 실행", type="primary", use_container_width=True)

with right:
    st.subheader("선수 프로필")

    if selected_player_display is None:
        st.info("선수를 선택하면 profile이 표시됩니다.")
    else:
        selected_raw_name = get_selected_raw_player_name(player_options, selected_player_display)
        row = get_profile_summary(player_profiles, selected_raw_name, canonical_player_key)

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
            )
            st.dataframe(profile_table, hide_index=True, use_container_width=True)

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

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
            stand=None,                # 선수 profile에서 자동 결정
            p_throws=p_throws,
            attack_angle=None,         # 선수 profile에서 자동 적용
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

    st.divider()
    st.subheader("예측 결과")

    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted distance", f"{pred_distance:.1f} ft")
    m2.metric("Model type", str(row["model_type"]))
    m3.metric("Swing-launch gap", f"{float(row['swing_launch_gap']):.2f}°")

    if warning:
        st.warning(warning)

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
        ],
        columns=["항목", "값"],
    )
    st.dataframe(result_table, hide_index=True, use_container_width=True)

    st.caption(
        "현재 예측은 타구속도, 발사각, 방향, 선수별 attack angle, 타석 방향, 투수 손 정보를 사용합니다. "
        "구장 정보는 아직 예측에 반영하지 않았습니다."
    )
