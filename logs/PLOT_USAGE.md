# plot_log.py 사용법

## 1. 단일 CSV plot

```bash
# 최신 CSV 자동 선택
python3 plot_log.py

# 특정 CSV 지정
python3 plot_log.py --csv mujoco_260402_135907.csv
```

출력: `logs/plot/<csv파일명>/`

| 파일 | 내용 |
|------|------|
| `01_imu_base_state.png` | 쿼터니언, ang vel, proj gravity, cmd vel, gait, value |
| `02_joint_pos_raw.png` | 관절 위치 raw (13DOF) |
| `03_joint_vel.png` | 관절 속도 (13DOF) |
| `04_actions.png` | RL action (12DOF) |
| `05_joint_pos_vs_action.png` | 관절 위치 vs action 오버레이 |
| `06_torque_joint_vs_motor.png` | 토크: joint space vs motor/4-bar |
| `07_obs_frame_47d.png` | 네트워크 입력 47D 의미별 분류 |

## 2. Sim vs Real 비교

```bash
python3 plot_log.py --compare mujoco_xxx.csv realrobot_xxx.csv
```

- 순서 무관 (자동으로 sim=blue, real=red 분류)
- 파일명만 넣어도 됨 (확장자, 경로 자동 탐색)

출력: `logs/plot/compare_<sim>_vs_<real>/`

| 파일 | 내용 |
|------|------|
| `00_compare_rmse_summary.png` | 신호 그룹별 RMSE 바 차트 (gap 요약) |
| `01_compare_imu_base.png` | IMU/base 오버레이 |
| `02~05_compare_*.png` | 관절 pos/vel/action 오버레이 |
| `06_compare_torque.png` | 토크 오버레이 |
| `07_compare_obs_frame.png` | 47D obs 전체 오버레이 |
| `08_compare_*.png` | 관절별 obs 개별 오버레이 |

## 3. 자동 감시 모드

```bash
# 백그라운드로 실행 - 새 CSV 생성 시 자동 plot
python3 plot_log.py --watch &
```

## 옵션

| 옵션 | 설명 |
|------|------|
| `--csv <path>` | 단일 CSV 지정 |
| `--compare <csv1> <csv2>` | sim vs real 비교 |
| `--watch` | 새 CSV 자동 감시 |
| `--show` | plot 창도 열기 |
