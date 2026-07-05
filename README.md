# Batted Ball Distance & Home Run Simulator

타구속도, 발사각, 타구 방향, 선수 정보, 투수 손 정보를 입력하면 예상 비거리와 구장별 홈런 여부를 예측하는 Streamlit 기반 시뮬레이터입니다.

본 프로젝트는 Statcast 타구 데이터를 기반으로 XGBoost 회귀 모델을 학습하고, 예측된 비거리를 MLB 구장별 펜스 거리·높이 데이터와 결합해 홈런 여부를 판정합니다.

---

## 1. 프로젝트 개요

일반적인 타구 비거리 예측은 `Exit Velocity`와 `Launch Angle` 중심으로 이루어집니다.  
본 프로젝트는 여기에 다음 요소를 추가했습니다.

- 타구 방향 5구간
- 타자 좌우 타석 정보
- 투수 손 정보
- 선수별 평균 Attack Angle
- `swing_launch_gap = launch_angle - attack_angle`
- 구장별 펜스 거리 및 높이
- late-drop 타구 궤적 기반 홈런 판정

최종 목표는 단순 비거리 예측이 아니라, 입력한 타구가 특정 구장에서 홈런이 되는지 확인하는 것입니다.

---

## 2. 사용 데이터

### 2.1 타구 데이터

모델 학습에는 MLB Statcast 기반 타구 데이터를 사용했습니다.

주요 사용 변수는 다음과 같습니다.

| 변수 | 설명 |
|---|---|
| `launch_speed` | 타구속도, mph |
| `launch_angle` | 발사각, deg |
| `hc_x`, `hc_y` | 타구 좌표 기반 spray angle 계산용 |
| `stand` | 타자 좌우 타석 |
| `p_throws` | 투수 손 |
| `attack_angle` | 스윙 궤도 각도 |
| `hit_distance_sc` | Statcast projected distance, 예측 대상 |

Coors Field와 Oracle Park는 구장 특수성이 크다고 판단해 학습 및 평가 데이터에서 제외했습니다.

### 2.2 구장 펜스 데이터

홈런 판정에는 `park_fence_data_28parks_5zones_hr.csv`를 사용했습니다.

각 구장은 5개 타구 방향으로 단순화했습니다.

| 방향 | 설명 |
|---|---|
| `left_line` | 좌측 라인 |
| `left_gap` | 좌중간 |
| `center` | 중앙 |
| `right_gap` | 우중간 |
| `right_line` | 우측 라인 |

각 방향별로 다음 값을 저장했습니다.

| 변수 | 설명 |
|---|---|
| `park` | 구장명 |
| `spray_zone5` | 타구 방향 |
| `fence_distance_ft` | 해당 방향의 펜스까지 거리 |
| `fence_height_ft` | 해당 방향의 펜스 높이 |

구장 치수는 Seamheads Ballparks Database, Baseball Savant Park Factors, MLB.com 공개 자료를 참고했습니다.

---

## 3. 모델 구조

### 3.1 비거리 예측 모델

비거리 예측에는 XGBoost 회귀 모델을 사용했습니다.

라인드라이브와 플라이볼은 타구 성격이 다르기 때문에 모델을 분리했습니다.

| 모델 | 발사각 기준 |
|---|---|
| `line_drive` | 10도 이상 25도 미만 |
| `fly_ball` | 25도 이상 50도 이하 |

입력된 발사각에 따라 앱이 자동으로 사용할 모델을 선택합니다.

### 3.2 주요 Feature

#### 수치형 Feature

| 변수 | 설명 |
|---|---|
| `launch_speed` | 타구속도 |
| `launch_angle` | 발사각 |
| `spray_angle` | 타구 방향을 각도로 변환한 값 |
| `R_ideal_ft` | 공기저항이 없다고 가정한 이상 포물선 비거리 |
| `attack_angle` | 선수별 평균 attack angle |
| `swing_launch_gap` | `launch_angle - attack_angle` |
| `abs_swing_launch_gap` | `swing_launch_gap`의 절댓값 |

#### 범주형 Feature

| 변수 | 설명 |
|---|---|
| `stand` | 타석 방향 |
| `p_throws` | 투수 손 |
| `matchup_side` | same-hand / opposite-hand |
| `spray_zone5` | 타구 방향 5구간 |
| `batted_direction5` | pull / oppo / center 방향 |
| `launch_angle_bin` | 발사각 구간 |

---

## 4. 선수 입력 처리

사용자는 선수 이름을 입력합니다.

앱은 학습 데이터에서 해당 선수의 profile을 찾아 다음 값을 자동으로 적용합니다.

- 선수별 평균 attack angle
- 타자의 기본 batting stand
- 스위치히터 여부
- 투수 손에 따른 좌우 타석

스위치히터의 경우 `p_throws`를 기준으로 타석 방향을 결정합니다.

| 투수 손 | 스위치히터 타석 |
|---|---|
| RHP | 좌타석 |
| LHP | 우타석 |

선수별 attack angle은 표본 수가 적을 경우 리그 평균 방향으로 shrinkage를 적용했습니다.

---

## 5. 테스트 성능

테스트 데이터 기준 성능은 다음과 같습니다.

### 전체 성능

| Metric | Value |
|---|---:|
| Test rows | 7,364 |
| RMSE | 16.60 ft |
| MAE | 12.34 ft |
| R² | 0.938 |
| Median absolute error | 9.34 ft |
| P90 absolute error | 27.02 ft |

### 모델별 성능

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| Fly ball | 13.81 ft | 10.30 ft | 0.938 |
| Line drive | 19.30 ft | 14.67 ft | 0.923 |

라인드라이브는 같은 타구속도와 발사각에서도 실제 비거리 편차가 크기 때문에 플라이볼보다 오차가 크게 나타났습니다.

---

## 6. 홈런 판정 원리

홈런 여부는 단순히 예측 비거리가 펜스 거리보다 긴지만 보지 않습니다.

앱은 다음 순서로 홈런 여부를 판정합니다.

1. XGBoost 모델로 예상 비거리 계산
2. 선택한 구장과 타구 방향의 펜스 거리·높이 조회
3. 예측 비거리에 맞는 late-drop 궤적 생성
4. 펜스 위치에서 공의 높이 계산
5. 공 높이가 펜스 높이를 넘으면 홈런으로 판정

### 판정 결과

| 결과 | 의미 |
|---|---|
| `HOME_RUN` | 펜스를 충분히 넘김 |
| `BORDERLINE_HR` | 펜스를 넘지만 여유가 작음 |
| `BORDERLINE_NOT_HR` | 펜스 근처에서 약간 부족 |
| `NOT_HR` | 홈런 아님 |

### Late-drop 궤적

실제 타구는 완전한 포물선처럼 움직이지 않습니다.  
초반에는 공이 뻗다가 후반에 급격히 떨어지는 경우가 많습니다.

이를 반영하기 위해 앱은 단순 포물선 대신 late-drop 형상 모델을 사용합니다.

해당 궤적은 다음 조건을 만족합니다.

- 타격 순간 높이에서 출발
- 입력한 launch angle 방향으로 출발
- XGBoost 예측 비거리 지점에서 착지
- 착지 직전에는 더 가파르게 낙하

이 방식은 실제 3D 물리 시뮬레이션은 아니지만, 예측 비거리와 펜스 위치를 연결하기 위한 후처리 모델입니다.

---

## 7. 실행 방법

### 7.1 필요한 파일

GitHub 저장소 구조는 아래처럼 두면 됩니다.

```text
Ohtani/
├─ app.py
├─ requirements.txt
├─ split_xgb_batted_distance.joblib
└─ park_fence_data_28parks_5zones_hr.csv
```

### 7.2 패키지 설치

```bash
pip install -r requirements.txt
```

`requirements.txt` 예시는 아래와 같습니다.

```txt
streamlit
pandas
numpy
joblib
xgboost
scikit-learn
pyarrow
```

### 7.3 앱 실행

```bash
streamlit run app.py
```

또는 Python 경로 문제가 있을 경우 아래처럼 실행할 수 있습니다.

```bash
python -m streamlit run app.py
```

---

## 8. 사용 방법

앱에서 다음 값을 입력합니다.

| 입력값 | 설명 |
|---|---|
| 선수 이름 | 예: Aaron Judge |
| Throwing hand | RHP 또는 LHP |
| Exit Velocity | 타구속도, mph |
| Launch Angle | 발사각, deg |
| 타구 방향 | 좌측 라인, 좌중간, 중앙, 우중간, 우측 라인 |
| 구장 | 홈런 판정 대상 구장 |

입력 후 `예측 실행` 버튼을 누르면 다음 값이 출력됩니다.

- 예상 비거리
- 모델 타입
- swing-launch gap
- 선택 구장 홈런 여부
- 펜스 거리
- 펜스 높이
- 펜스 위치에서 공 높이
- 전체 구장별 홈런 여부 비교

---

## 9. Streamlit Cloud 배포

Streamlit Community Cloud에 배포할 경우 다음 설정을 사용합니다.

| 항목 | 값 |
|---|---|
| Repository | 현재 GitHub 저장소 |
| Branch | `main` |
| Main file path | `app.py` |

배포 후 생성된 Streamlit 링크를 공유하면 다른 사용자도 웹에서 실행할 수 있습니다.

---

## 10. 한계점

이 모델은 실제 타구 궤적을 완전히 복원하는 모델은 아닙니다.

주요 한계는 다음과 같습니다.

- 실제 타구 스핀량과 스핀축을 직접 사용하지 않음
- 바람, 온도, 습도, 고도 효과를 직접 반영하지 않음
- 타구의 좌우 휘어짐, hook/slice를 반영하지 않음
- 구장 펜스를 5개 방향으로 단순화함
- 펜스 위 노란선, 관중석 구조, 파울폴 근처 판정은 단순화됨
- `hit_distance_sc` 자체가 Statcast projected distance이므로 실제 낙하지점과 다를 수 있음

따라서 본 앱은 실제 경기 판정용이 아니라, 타구 조건과 구장 구조에 따른 홈런 가능성을 비교하는 분석용 시뮬레이터입니다.

---

## 11. 프로젝트 요약

본 프로젝트는 Statcast 타구 데이터를 기반으로 타구 비거리를 예측하고, 구장별 펜스 구조를 결합해 홈런 여부를 판정하는 시뮬레이터입니다.

단순히 타구 비거리를 예측하는 데서 끝나지 않고, 선수별 swing profile, 투수 손, 타구 방향, 구장별 펜스 조건을 연결해 실제 야구장 환경에 가까운 해석을 시도했습니다.
