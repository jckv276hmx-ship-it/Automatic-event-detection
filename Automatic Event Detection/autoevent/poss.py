from dataclasses import dataclass
from fnmatch import fnmatch

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from tools.config import EPS_S, EPS_THETA, EPS_V, R_PZ, R_DZ, SG_WINDOW, SG_POLYORDER

from fnmatch import fnmatch

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from tools.config import (
    R_PZ,
    R_DZ,
    EPS_S,
    EPS_THETA,
    EPS_V,
    SG_WINDOW,
    SG_POLYORDER,
)

def get_players(trk: pd.DataFrame) -> list[str]:
    x_cols = [
        col for col in trk.columns
        if fnmatch(col, "home_*_x") or fnmatch(col, "away_*_x")
    ]
    return sorted({col[:-2] for col in x_cols})


class PossessionDetector:
    def __init__(self, tracking: pd.DataFrame):
        self.tracking = tracking.copy()
        self.players = get_players(self.tracking)

    def run(self):
        return (
            self.smooth_ball()
            .add_ball_kinematics()
            .add_player_ball_distances()
            .add_ball_control()
            .add_control_sequences()
            .add_possession_losses()
            .add_possession_gains()
            .tracking
        )

    # Savitzy-Golay filter로 볼 위치 smoothing (이후 속도 계산에 사용)
    def smooth_ball(self):
        alive_mask = self.tracking["ball_state"].eq("alive")
        self.tracking.loc[~alive_mask, ["ball_x", "ball_y"]] = np.nan

        # alive / dead가 바뀔 때마다 새로운 구간 id 부여
        episode_id = alive_mask.ne(alive_mask.shift(fill_value=False)).cumsum()

        for seg in episode_id[alive_mask].unique():
            idx = self.tracking.index[(episode_id == seg) & alive_mask]
            xy = self.tracking.loc[idx, ["ball_x", "ball_y"]]

            if len(xy) == 0:
                continue

            window = min(SG_WINDOW, len(xy))
            if window % 2 == 0:
                window -= 1

            # 너무 짧은 구간은 smoothing하지 않고 그대로 둠
            if window < 3 or len(xy) < SG_POLYORDER + 2:
                continue

            self.tracking.loc[idx, "ball_x"] = savgol_filter(
                xy["ball_x"].to_numpy(),
                window_length=window,
                polyorder=min(SG_POLYORDER, window - 1),
                mode="interp",
            )
            self.tracking.loc[idx, "ball_y"] = savgol_filter(
                xy["ball_y"].to_numpy(),
                window_length=window,
                polyorder=min(SG_POLYORDER, window - 1),
                mode="interp",
            )

        return self    

    def add_ball_kinematics(self):

        self.tracking["ball_dx_prev"] = self.tracking["ball_x"] - self.tracking["ball_x"].shift(1)
        self.tracking["ball_dy_prev"] = self.tracking["ball_y"] - self.tracking["ball_y"].shift(1)

        self.tracking["ball_dx_next"] = self.tracking["ball_x"].shift(-1) - self.tracking["ball_x"]
        self.tracking["ball_dy_next"] = self.tracking["ball_y"].shift(-1) - self.tracking["ball_y"]

        self.tracking["ball_speed_prev"] = np.sqrt(
            self.tracking["ball_dx_prev"] ** 2 + self.tracking["ball_dy_prev"] ** 2
        )
        self.tracking["ball_speed_next"] = np.sqrt(
            self.tracking["ball_dx_next"] ** 2 + self.tracking["ball_dy_next"] ** 2
        )

        prev_nonzero = self.tracking["ball_speed_prev"].replace(0, np.nan)
        next_nonzero = self.tracking["ball_speed_next"].replace(0, np.nan)

        self.tracking["ball_dir_in_x"] = self.tracking["ball_dx_prev"] / prev_nonzero
        self.tracking["ball_dir_in_y"] = self.tracking["ball_dy_prev"] / prev_nonzero

        self.tracking["ball_dir_out_x"] = self.tracking["ball_dx_next"] / next_nonzero
        self.tracking["ball_dir_out_y"] = self.tracking["ball_dy_next"] / next_nonzero

        # 공의 이동량
        self.tracking["ball_displacement"] = self.tracking["ball_speed_next"]

        return self
    
    def add_player_ball_distances(self):
        for player in self.players:
            self.tracking[f"dist_{player}"] = np.sqrt(
                (self.tracking[f"{player}_x"] - self.tracking["ball_x"]) ** 2
                + (self.tracking[f"{player}_y"] - self.tracking["ball_y"]) ** 2
            )
        return self

    def add_ball_control(self):
        self.tracking["ball_control"] = "no_possession"
        self.tracking["controller_id"] = pd.NA
        self.tracking["controller_team"] = pd.NA
        self.tracking["controller_distance"] = np.nan
        self.tracking["duel_players"] = pd.NA

        for idx, row in self.tracking.iterrows():
            if row["ball_state"] == "dead":
                self.tracking.at[idx, "ball_control"] = "dead_ball"
                continue

            home_in_pz, away_in_pz = [], []
            home_in_dz, away_in_dz = [], []

            for player in self.players:
                dist = row[f"dist_{player}"]
                if pd.isna(dist):
                    continue

                is_home = player.startswith("home_")

                if dist <= R_PZ:
                    (home_in_pz if is_home else away_in_pz).append((player, dist))

                if dist <= R_DZ:
                    (home_in_dz if is_home else away_in_dz).append((player, dist))

            if home_in_dz and away_in_dz:
                duel_candidates = sorted(home_in_dz + away_in_dz, key=lambda x: x[1])
                self.tracking.at[idx, "ball_control"] = "duel"
                self.tracking.at[idx, "duel_players"] = "|".join(player for player, _ in duel_candidates)
                continue

            possession_candidates = home_in_pz + away_in_pz
            if possession_candidates:
                controller_id, controller_distance = min(possession_candidates, key=lambda x: x[1])
                self.tracking.at[idx, "ball_control"] = "possession"
                self.tracking.at[idx, "controller_id"] = controller_id
                self.tracking.at[idx, "controller_team"] = controller_id.split("_")[0]
                self.tracking.at[idx, "controller_distance"] = controller_distance

        return self

    def add_control_sequences(self):
        self.tracking["control_sequence_id"] = pd.NA
        self.tracking["control_sequence_type"] = pd.NA
        self.tracking["control_sequence_player"] = pd.NA

        seq_id = 0
        prev_type = None
        prev_player = None

        for idx, row in self.tracking.iterrows():
            cur_type = row["ball_control"]

            if cur_type not in ["possession", "duel"]:
                prev_type = None
                prev_player = None
                continue

            if cur_type == "possession":
                cur_player = row["controller_id"]
                same_sequence = (prev_type == "possession") and (prev_player == cur_player)
            else:
                cur_player = pd.NA
                same_sequence = (prev_type == "duel")

            if not same_sequence:
                seq_id += 1

            self.tracking.at[idx, "control_sequence_id"] = seq_id
            self.tracking.at[idx, "control_sequence_type"] = cur_type
            self.tracking.at[idx, "control_sequence_player"] = cur_player

            prev_type = cur_type
            prev_player = cur_player

        return self
    
    def add_possession_losses(self):
        self.tracking["is_loss"] = False
        self.tracking["loss_player"] = pd.NA
        self.tracking["loss_team"] = pd.NA

        for i in range(len(self.tracking) - 1):
            row = self.tracking.iloc[i]

            if row["ball_control"] != "possession": # duel도 제외함.
                continue

            player = row["controller_id"]
            if pd.isna(player):
                continue

            next_dist = self.tracking.iloc[i + 1][f"dist_{player}"]
            ball_displacement = row["ball_displacement"]

            # 조건 1번
            # the ball is outside the PZ of player A at frame f+1
            # the ball displacement is above a given threshold EPS_S
            outside_pz = pd.isna(next_dist) or (next_dist > R_PZ)
            enough_movement = (not pd.isna(ball_displacement)) and (ball_displacement > EPS_S)

            if not (outside_pz and enough_movement):
                continue

            next_control_idx = None
            for j in range(i + 1, len(self.tracking)):
                if self.tracking.iloc[j]["ball_control"] in ["possession", "duel"]:
                    next_control_idx = j
                    break

            if next_control_idx is None:
                self.tracking.at[self.tracking.index[i], "is_loss"] = True
                self.tracking.at[self.tracking.index[i], "loss_player"] = player
                self.tracking.at[self.tracking.index[i], "loss_team"] = player.split("_")[0]
                continue

            next_row = self.tracking.iloc[next_control_idx]

            if next_row["ball_control"] == "possession" and next_row["controller_id"] == player:
                continue

            self.tracking.at[self.tracking.index[i], "is_loss"] = True
            self.tracking.at[self.tracking.index[i], "loss_player"] = player
            self.tracking.at[self.tracking.index[i], "loss_team"] = player.split("_")[0]

        return self
    
    def add_possession_gains(self):
        """
        gain 규칙:
        같은 선수의 control sequence [f0, ..., fn]에 대해
        - 시작의 incoming direction과 끝의 outgoing direction이 충분히 다르거나
        - sequence 내부 어떤 frame에서든 speed 변화가 충분히 크면
        gain을 sequence 시작점 f0에 기록
        """
                
        self.tracking["is_gain"] = False
        self.tracking["gain_player"] = pd.NA
        self.tracking["gain_team"] = pd.NA

        sequence_ids = self.tracking["control_sequence_id"].dropna().unique()

        for seq_id in sequence_ids:
            seq = self.tracking[self.tracking["control_sequence_id"] == seq_id]

            if seq.empty:
                continue

            if seq["control_sequence_type"].iloc[0] != "possession":
                continue

            player = seq["control_sequence_player"].iloc[0]
            if pd.isna(player):
                continue

            start_idx = seq.index[0]
            end_idx = seq.index[-1]

            start_row = self.tracking.loc[start_idx]
            end_row = self.tracking.loc[end_idx]

            has_direction_change = False
            has_speed_change = False

            if not (
                pd.isna(start_row["ball_dir_in_x"]) or
                pd.isna(start_row["ball_dir_in_y"]) or
                pd.isna(end_row["ball_dir_out_x"]) or
                pd.isna(end_row["ball_dir_out_y"])
            ):
                dot_product = (
                    start_row["ball_dir_in_x"] * end_row["ball_dir_out_x"] +
                    start_row["ball_dir_in_y"] * end_row["ball_dir_out_y"]
                )
                if dot_product < EPS_THETA:
                    has_direction_change = True

            valid_speed = seq[["ball_speed_prev", "ball_speed_next"]].dropna()
            if not valid_speed.empty:
                speed_diff = np.abs(
                    valid_speed["ball_speed_next"] - valid_speed["ball_speed_prev"]
                )
                if np.any(speed_diff > EPS_V):
                    has_speed_change = True

            if has_direction_change or has_speed_change:
                self.tracking.at[start_idx, "is_gain"] = True
                self.tracking.at[start_idx, "gain_player"] = player
                self.tracking.at[start_idx, "gain_team"] = player.split("_")[0]

        return self