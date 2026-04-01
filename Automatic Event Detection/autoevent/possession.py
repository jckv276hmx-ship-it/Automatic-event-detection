from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from tools import config


# =========================================================
# Config
# =========================================================

@dataclass
class PossessionConfig:
    # provider B/C baseline from paper-like setting
    r_pz: float = config.R_PZ
    r_dz: float = config.R_DZ

    # loss / gain thresholds
    eps_s: float = config.EPS_S
    eps_theta: float = config.EPS_THETA   # dot product threshold
    eps_v: float = config.EPS_V

    # smoothing
    sg_window: int = config.SG_WINDOW
    sg_polyorder: int = config.SG_POLYORDER


# =========================================================
# Schema helper
# =========================================================

class TrackingSchema:
    """tracking DataFrame의 컬럼 이름과 선수 목록을 관리하는 헬퍼."""

    def __init__(self, trk: pd.DataFrame):
        self.trk = trk

    @property
    def ball_x(self) -> str:
        return "ball_x_smooth" if "ball_x_smooth" in self.trk.columns else "ball_x"

    @property
    def ball_y(self) -> str:
        return "ball_y_smooth" if "ball_y_smooth" in self.trk.columns else "ball_y"

    @property
    def players(self) -> list[str]:
        x_cols = [
            c for c in self.trk.columns
            if fnmatch(c, "home_*_x") or fnmatch(c, "away_*_x")
        ]
        return sorted({c[:-2] for c in x_cols})

    @staticmethod
    def team_of(player_id: str) -> str:
        return str(player_id).split("_")[0]

    @staticmethod
    def dist_col(player_id: str) -> str:
        return f"dist_{player_id}"

    @staticmethod
    def x_col(player_id: str) -> str:
        return f"{player_id}_x"

    @staticmethod
    def y_col(player_id: str) -> str:
        return f"{player_id}_y"


# =========================================================
# Possession detector
# =========================================================

class PossessionDetector:
    """
    Open-play possession step detector.

    현재 단계:
    - ball smoothing
    - ball kinematics
    - player-ball distance
    - ball control (dead_ball / no_possession / possession / duel)
    - open-play possession loss
    - open-play possession gain

    세트피스 restart gain은 나중에 별도 로직으로 추가하는 것을 권장.
    """

    def __init__(self, tracking: pd.DataFrame, cfg: PossessionConfig | None = None):
        self.trk = tracking.copy()
        self.cfg = cfg or PossessionConfig()
        self.schema = TrackingSchema(self.trk)

    # -----------------------------------------------------
    # public pipeline
    # -----------------------------------------------------

    def run(self) -> pd.DataFrame:
        return (
            self.smooth_ball()
            .add_ball_kinematics()
            .add_player_ball_distances()
            .add_ball_control()
            .add_control_sequence_ids()
            .add_open_play_losses()
            .add_open_play_gains()
            .trk
        )

    # -----------------------------------------------------
    # step 1: smooth ball
    # -----------------------------------------------------

    def smooth_ball(self) -> "PossessionDetector":
        """
        ball_state == 'alive' 인 연속 구간별로만 Savitzky-Golay smoothing.
        dead 구간을 건너뛰며 이어붙여 smoothing하지 않음.
        """
        self.trk["ball_x_smooth"] = np.nan
        self.trk["ball_y_smooth"] = np.nan

        alive_mask = self.trk["ball_state"].eq("alive")

        # alive 연속 구간 id 만들기
        # alive 여부가 바뀔 때마다 구간이 달라짐
        segment_id = alive_mask.ne(alive_mask.shift(fill_value=False)).cumsum()

        for seg in segment_id[alive_mask].unique():
            idx = self.trk.index[(segment_id == seg) & alive_mask]
            sub = self.trk.loc[idx, ["ball_x", "ball_y"]]

            if len(sub) == 0:
                continue

            # window는 홀수여야 함
            window = min(self.cfg.sg_window, len(sub))
            if window % 2 == 0:
                window -= 1

            # 너무 짧으면 원본 사용
            if window < 3 or len(sub) < self.cfg.sg_polyorder + 2:
                self.trk.loc[idx, "ball_x_smooth"] = sub["ball_x"].to_numpy()
                self.trk.loc[idx, "ball_y_smooth"] = sub["ball_y"].to_numpy()
                continue

            self.trk.loc[idx, "ball_x_smooth"] = savgol_filter(
                sub["ball_x"].to_numpy(),
                window_length=window,
                polyorder=min(self.cfg.sg_polyorder, window - 1),
                mode="interp",
            )
            self.trk.loc[idx, "ball_y_smooth"] = savgol_filter(
                sub["ball_y"].to_numpy(),
                window_length=window,
                polyorder=min(self.cfg.sg_polyorder, window - 1),
                mode="interp",
            )

        return self

    # -----------------------------------------------------
    # step 2: ball kinematics
    # -----------------------------------------------------

    def add_ball_kinematics(self) -> "PossessionDetector":
        """
        논문식 gain/loss 판정을 위해
        - next displacement / speed / outgoing direction
        - prev displacement / speed / incoming direction
        를 모두 계산
        """
        bx, by = self.schema.ball_x, self.schema.ball_y


        # next
        self.trk["ball_dx_next"] = self.trk[bx].shift(-1) - self.trk[bx]
        self.trk["ball_dy_next"] = self.trk[by].shift(-1) - self.trk[by]
        self.trk["ball_displacement_next"] = np.sqrt(
            self.trk["ball_dx_next"] ** 2 + self.trk["ball_dy_next"] ** 2
        )
        self.trk["ball_speed_next"] = self.trk["ball_displacement_next"]

        # prev
        self.trk["ball_dx_prev"] = self.trk[bx] - self.trk[bx].shift(1)
        self.trk["ball_dy_prev"] = self.trk[by] - self.trk[by].shift(1)
        self.trk["ball_displacement_prev"] = np.sqrt(
            self.trk["ball_dx_prev"] ** 2 + self.trk["ball_dy_prev"] ** 2
        )
        self.trk["ball_speed_prev"] = self.trk["ball_displacement_prev"]

        # normalize directions
        next_nonzero = self.trk["ball_speed_next"].replace(0, np.nan)
        prev_nonzero = self.trk["ball_speed_prev"].replace(0, np.nan)

        # outgoing direction at frame f : f -> f+1
        self.trk["ball_dir_out_x"] = self.trk["ball_dx_next"] / next_nonzero
        self.trk["ball_dir_out_y"] = self.trk["ball_dy_next"] / next_nonzero

        # incoming direction at frame f : f-1 -> f
        self.trk["ball_dir_in_x"] = self.trk["ball_dx_prev"] / prev_nonzero
        self.trk["ball_dir_in_y"] = self.trk["ball_dy_prev"] / prev_nonzero

        return self

    # -----------------------------------------------------
    # step 3: player-ball distances
    # -----------------------------------------------------

    def add_player_ball_distances(self) -> "PossessionDetector":
        bx, by = self.schema.ball_x, self.schema.ball_y

        for p in self.schema.players:
            self.trk[self.schema.dist_col(p)] = np.sqrt(
                (self.trk[self.schema.x_col(p)] - self.trk[bx]) ** 2 +
                (self.trk[self.schema.y_col(p)] - self.trk[by]) ** 2
            )

        return self

    # -----------------------------------------------------
    # step 4: ball control
    # -----------------------------------------------------

    def add_ball_control(self) -> "PossessionDetector":
        """
        frame 단위 ball control 분류:
        - dead_ball
        - no_possession
        - possession
        - duel

        duel은 양 팀 선수 모두 DZ 안에 있으면 우선.
        """
        self.trk["ball_control"] = "no_possession"
        self.trk["controller_id"] = pd.NA
        self.trk["controller_team"] = pd.NA
        self.trk["controller_distance"] = np.nan
        self.trk["duel_players"] = pd.NA

        r_pz = self.cfg.r_pz
        r_dz = self.cfg.r_dz

        dist_cols = [self.schema.dist_col(p) for p in self.schema.players]

        for idx, row in self.trk.iterrows():
            if row["ball_state"] == "dead":
                self.trk.at[idx, "ball_control"] = "dead_ball"
                continue

            home_in_pz, away_in_pz = [], []
            home_in_dz, away_in_dz = [], []

            for col in dist_cols:
                d = row[col]
                if pd.isna(d):
                    continue

                pid = col.replace("dist_", "")
                is_home = pid.startswith("home_")

                if d <= r_pz:
                    (home_in_pz if is_home else away_in_pz).append((pid, d))

                if d <= r_dz:
                    (home_in_dz if is_home else away_in_dz).append((pid, d))

            # duel 우선
            if home_in_dz and away_in_dz:
                duel_candidates = sorted(home_in_dz + away_in_dz, key=lambda x: x[1])
                self.trk.at[idx, "ball_control"] = "duel"
                self.trk.at[idx, "duel_players"] = "|".join(pid for pid, _ in duel_candidates)
                continue

            # possession
            possession_candidates = home_in_pz + away_in_pz
            if possession_candidates:
                controller_id, controller_distance = min(possession_candidates, key=lambda x: x[1])

                self.trk.at[idx, "ball_control"] = "possession"
                self.trk.at[idx, "controller_id"] = controller_id
                self.trk.at[idx, "controller_team"] = self.schema.team_of(controller_id)
                self.trk.at[idx, "controller_distance"] = controller_distance
            else:
                self.trk.at[idx, "ball_control"] = "no_possession"

        return self

    # -----------------------------------------------------
    # step 5: control sequence ids
    # -----------------------------------------------------

    def add_control_sequence_ids(self) -> "PossessionDetector":
        """
        같은 controller_id의 연속 possession 구간을 묶음.
        non-possession / duel / dead_ball은 sequence 없음.
        """
        self.trk["control_sequence_id"] = pd.NA

        seq_id = 0
        prev_controller = None
        prev_control = None

        for i in range(len(self.trk)):
            control = self.trk.iloc[i]["ball_control"]
            controller = self.trk.iloc[i]["controller_id"]

            if control != "possession" or pd.isna(controller):
                prev_controller = None
                prev_control = control
                continue

            if prev_control == "possession" and controller == prev_controller:
                # 같은 sequence 유지
                self.trk.at[self.trk.index[i], "control_sequence_id"] = seq_id
            else:
                seq_id += 1
                self.trk.at[self.trk.index[i], "control_sequence_id"] = seq_id

            prev_controller = controller
            prev_control = control

        return self

    # -----------------------------------------------------
    # step 6: losses
    # -----------------------------------------------------

    def add_open_play_losses(self) -> "PossessionDetector":
        """
        loss 규칙:
        1) 다음 프레임에 공이 그 선수 PZ 밖
        2) ball displacement 충분히 큼
        3) 다음 control frame에서 같은 선수가 다시 나오지 않음
        """
        self.trk["is_loss"] = False
        self.trk["loss_player"] = pd.NA
        self.trk["loss_team"] = pd.NA

        for i in range(len(self.trk) - 1):
            row = self.trk.iloc[i]

            if row["ball_control"] != "possession" or pd.isna(row["controller_id"]):
                continue

            pid = row["controller_id"]
            dist_col = self.schema.dist_col(pid)

            if dist_col not in self.trk.columns:
                continue

            next_row = self.trk.iloc[i + 1]
            next_distance = next_row[dist_col]
            displacement = row["ball_displacement_next"]

            outside_pz = pd.isna(next_distance) or (next_distance > self.cfg.r_pz)
            enough_ball_movement = (not pd.isna(displacement)) and (displacement > self.cfg.eps_s)

            if not (outside_pz and enough_ball_movement):
                continue

            next_control_idx = self._find_next_control_frame(i + 1)

            # 이후 control frame이 없으면 loss로 볼 수 있음
            if next_control_idx is None:
                self.trk.at[self.trk.index[i], "is_loss"] = True
                self.trk.at[self.trk.index[i], "loss_player"] = pid
                self.trk.at[self.trk.index[i], "loss_team"] = self.schema.team_of(pid)
                continue

            next_controller = self.trk.iloc[next_control_idx]["controller_id"]

            # 다음 control frame에서 같은 선수가 다시 나오면 loss 아님
            if pd.isna(next_controller) or next_controller != pid:
                self.trk.at[self.trk.index[i], "is_loss"] = True
                self.trk.at[self.trk.index[i], "loss_player"] = pid
                self.trk.at[self.trk.index[i], "loss_team"] = self.schema.team_of(pid)

        return self

    # -----------------------------------------------------
    # step 7: gains
    # -----------------------------------------------------

    def add_open_play_gains(self) -> "PossessionDetector":
        """
        gain 규칙:
        같은 선수의 control sequence [f0, ..., fn]에 대해
        - 시작의 incoming direction과 끝의 outgoing direction이 충분히 다르거나
        - sequence 내부 어떤 frame에서든 speed 변화가 충분히 크면
        gain을 sequence 시작점 f0에 기록
        """
        self.trk["is_gain"] = False
        self.trk["gain_player"] = pd.NA
        self.trk["gain_team"] = pd.NA

        seq_ids = self.trk["control_sequence_id"].dropna().unique()

        for seq_id in seq_ids:
            seq = self.trk[self.trk["control_sequence_id"] == seq_id]
            if seq.empty:
                continue

            start_idx = seq.index[0]
            end_idx = seq.index[-1]
            start_row = self.trk.loc[start_idx]
            end_row = self.trk.loc[end_idx]

            pid = start_row["controller_id"]
            if pd.isna(pid):
                continue

            has_direction_change = self._sequence_has_direction_change(start_row, end_row)
            has_speed_change = self._sequence_has_speed_change(seq)

            if has_direction_change or has_speed_change:
                self.trk.at[start_idx, "is_gain"] = True
                self.trk.at[start_idx, "gain_player"] = pid
                self.trk.at[start_idx, "gain_team"] = self.schema.team_of(pid)

        return self

    # -----------------------------------------------------
    # private helpers
    # -----------------------------------------------------

    def _find_next_control_frame(self, start_pos: int) -> int | None:
        """
        start_pos 이후 처음으로 possession 또는 duel이 나타나는 frame의 positional index 반환.
        """
        for j in range(start_pos, len(self.trk)):
            control = self.trk.iloc[j]["ball_control"]
            if control in ("possession", "duel"):
                return j
        return None

    def _sequence_has_direction_change(self, start_row: pd.Series, end_row: pd.Series) -> bool:
        in_x = start_row["ball_dir_in_x"]
        in_y = start_row["ball_dir_in_y"]
        out_x = end_row["ball_dir_out_x"]
        out_y = end_row["ball_dir_out_y"]

        if pd.isna(in_x) or pd.isna(in_y) or pd.isna(out_x) or pd.isna(out_y):
            return False

        dot_product = in_x * out_x + in_y * out_y
        return bool(dot_product < self.cfg.eps_theta)

    def _sequence_has_speed_change(self, seq: pd.DataFrame) -> bool:
        """
        각 frame fi에서 |v_in(fi) - v_out(fi)| > eps_v 인 frame이 하나라도 있으면 True
        """
        vin = seq["ball_speed_prev"].to_numpy()
        vout = seq["ball_speed_next"].to_numpy()

        valid = ~(np.isnan(vin) | np.isnan(vout))
        if not np.any(valid):
            return False

        delta_v = np.abs(vin[valid] - vout[valid])
        return bool(np.any(delta_v > self.cfg.eps_v))


# =========================================================
# backward-compatible wrappers
# =========================================================

_DEFAULT_CFG = PossessionConfig()


def get_player_columns(tracking: pd.DataFrame) -> list[str]:
    return TrackingSchema(tracking).players


def smooth_ball(tracking: pd.DataFrame) -> pd.DataFrame:
    return PossessionDetector(tracking).smooth_ball().trk


def compute_ball_kinematics(tracking: pd.DataFrame) -> pd.DataFrame:
    return PossessionDetector(tracking).add_ball_kinematics().trk


def compute_player_ball_distances(tracking: pd.DataFrame) -> pd.DataFrame:
    return (
        PossessionDetector(tracking)
        .add_player_ball_distances()
        .trk
    )


def compute_ball_control(
    tracking: pd.DataFrame,
    r_pz: float = _DEFAULT_CFG.r_pz,
    r_dz: float = _DEFAULT_CFG.r_dz,
) -> pd.DataFrame:
    cfg = PossessionConfig(r_pz=r_pz, r_dz=r_dz)
    return (
        PossessionDetector(tracking, cfg)
        .add_ball_control()
        .trk
    )


def detect_possession_losses(
    tracking: pd.DataFrame,
    r_pz: float = _DEFAULT_CFG.r_pz,
    eps_s: float = _DEFAULT_CFG.eps_s,
) -> pd.DataFrame:
    cfg = PossessionConfig(r_pz=r_pz, eps_s=eps_s)
    return (
        PossessionDetector(tracking, cfg)
        .add_ball_kinematics()
        .add_open_play_losses()
        .trk
    )


def detect_possession_gains(
    tracking: pd.DataFrame,
    eps_theta: float = _DEFAULT_CFG.eps_theta,
    eps_v: float = _DEFAULT_CFG.eps_v,
) -> pd.DataFrame:
    cfg = PossessionConfig(eps_theta=eps_theta, eps_v=eps_v)
    return (
        PossessionDetector(tracking, cfg)
        .add_ball_kinematics()
        .add_control_sequence_ids()
        .add_open_play_gains()
        .trk
    )


def run_possession_pipeline(
    tracking: pd.DataFrame,
    r_pz: float = _DEFAULT_CFG.r_pz,
    r_dz: float = _DEFAULT_CFG.r_dz,
    eps_s: float = _DEFAULT_CFG.eps_s,
    eps_theta: float = _DEFAULT_CFG.eps_theta,
    eps_v: float = _DEFAULT_CFG.eps_v,
) -> pd.DataFrame:
    cfg = PossessionConfig(
        r_pz=r_pz,
        r_dz=r_dz,
        eps_s=eps_s,
        eps_theta=eps_theta,
        eps_v=eps_v,
    )
    return PossessionDetector(tracking, cfg).run()