from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from autoevent.helpers import team_of as _team_of_fn
from tools import config


@dataclass
class SetPieceConfig:
	r_pz: float = config.R_PZ
	kickoff_x_tol: float = config.KICKOFF_X_TOL
	kickoff_y_tol: float = config.KICKOFF_Y_TOL
	center_x: float = config.CENTER_X
	center_y: float = config.CENTER_Y
	pitch_y: float = config.PITCH_Y
	pitch_x: float = config.PITCH_X
	half_tol: float = config.HALF_TOL
	penalty_mark_x: float = config.PENALTY_MARK_X
	penalty_mark_tol: float = config.PENALTY_MARK_TOL
	penalty_gk_x_tol: float = config.PENALTY_GK_X_TOL
	penalty_gk_y_tol: float = config.PENALTY_GK_Y_TOL
	penalty_other_players_tol: float = config.PENALTY_OTHER_PLAYERS_TOL
	corner_tol: float = config.CORNER_TOL
	throw_in_tol: float = config.THROW_IN_TOL
	goal_area_x: float = config.GOAL_AREA_X
	goal_area_x_tol: float = config.GOAL_AREA_X_TOL
	goal_area_y_tol: float = config.GOAL_AREA_Y_TOL
	goal_area_y: float = config.GOAL_AREA_Y

	penalty_area_x_max: float = config.PENALTY_AREA_X_MAX
	penalty_area_y_min: float = config.PENALTY_AREA_Y_MIN
	penalty_area_y_max: float = config.PENALTY_AREA_Y_MAX

	fallback_scan_step: int = config.FALLBACK_SCAN_STEP
	sp_loss_map_window: float = config.SP_LOSS_MAP_WINDOW
	throw_in_ball_z_min: float = config.THROW_IN_BALL_Z_MIN


class SetPieceTrigger:

# Ball in Player's PZ
	def _ball_in_player_pz(self, row: pd.Series, player: str) -> bool:
		dist = row[f"dist_{player}"]
		if pd.isna(dist):
			return False

		return dist <= self.cfg.r_pz

# 킥오프 트리거
	def _get_kickoff_trigger_player(self, row: pd.Series, first_in_row: pd.Series) -> str | None:
		if not self._all_players_in_own_half(row):
			return None

		candidates: list[str] = []

		for player in self.players:
			player_x = row[self.x_col(player)]
			player_y = row[self.y_col(player)]

			if pd.isna(player_x) or pd.isna(player_y):
				continue

			# 논문에는 원형 범위로 표현되어 있음
			near_center = (
				abs(player_x - self.cfg.center_x) <= self.cfg.kickoff_x_tol
				and abs(player_y - self.cfg.center_y) <= self.cfg.kickoff_y_tol
			)
			if not near_center:
				continue

			candidates.append(player)

		return self._select_nearest_by_first_in_dist(first_in_row, candidates)
	
	def _all_players_in_own_half(self, row: pd.Series) -> bool:
		for player in self.players:
			player_x = row[self.x_col(player)]
			if pd.isna(player_x):
				continue

			if player.startswith("home_") and player_x > self.cfg.center_x + self.cfg.half_tol:
				return False

			if player.startswith("away_") and player_x < self.cfg.center_x - self.cfg.half_tol:
				return False

		return True

	
# 페널티킥 트리거
	def _get_penalty_trigger_player(self, row: pd.Series, first_in_row: pd.Series) -> str | None:
		candidates: list[str] = []

		for player in self.players:
			player_x = row[self.x_col(player)]
			player_y = row[self.y_col(player)]
			if pd.isna(player_x) or pd.isna(player_y):
				continue

			penalty_mark_x = self._penalty_mark_x(self.team_of(player))
			near_penalty_mark = (
				abs(player_x - penalty_mark_x) <= self.cfg.penalty_mark_tol
				and abs(player_y - self.cfg.center_y) <= self.cfg.penalty_mark_tol
			)
			if not near_penalty_mark:
				continue

			candidates.append(player)

		return self._select_nearest_by_first_in_dist(first_in_row, candidates)

	def _is_penalty_setup(self, row: pd.Series, executor: str) -> bool:
		executor_team = self.team_of(executor)
		opponent_team = "away" if executor_team == "home" else "home"
		penalty_mark_x = self._penalty_mark_x(executor_team)
		goal_x = self._goal_x(executor_team)
		penalty_box_x_min, penalty_box_x_max = self._penalty_box_x_bounds(executor_team)

		executor_x = row[self.x_col(executor)]
		executor_y = row[self.y_col(executor)]

		executor_on_mark = (
			abs(executor_x - penalty_mark_x) <= self.cfg.penalty_mark_tol
			and abs(executor_y - self.cfg.center_y) <= self.cfg.penalty_mark_tol
		)
		if not executor_on_mark:
			return False

		gk_candidates: list[str] = []
		for player in self.players:
			if self.team_of(player) != opponent_team:
				continue

			player_x = row[self.x_col(player)]
			player_y = row[self.y_col(player)]
			if pd.isna(player_x) or pd.isna(player_y):
				continue

			near_goal_line = abs(player_x - goal_x) <= self.cfg.penalty_gk_x_tol
			between_posts = abs(player_y - self.cfg.center_y) <= self.cfg.goal_area_y + self.cfg.penalty_gk_y_tol
			if near_goal_line and between_posts:
				gk_candidates.append(player)

		if len(gk_candidates) != 1:
			return False

		for player in self.players:
			if player == executor or player in gk_candidates:
				continue

			player_x = row[self.x_col(player)]
			player_y = row[self.y_col(player)]
			if pd.isna(player_x) or pd.isna(player_y):
				continue
			
			# 페널티 박스 근처에 다른 선수가 있으면 안 됨 (GK 제외)
			# min + tol, max - tol: 선수가 페널티 박스 경계에 붙어있는 경우를 허용하기 위해 tol만큼 여유를 둠
			# 추후에 penalty_arc 범위도 추가해 near_penalty_box 수정
			near_penalty_box = (
				penalty_box_x_min + self.cfg.penalty_other_players_tol <= player_x <= penalty_box_x_max - self.cfg.penalty_other_players_tol
				and self.cfg.penalty_area_y_min + self.cfg.penalty_other_players_tol <= player_y <= self.cfg.penalty_area_y_max - self.cfg.penalty_other_players_tol
			)
			if near_penalty_box:
				return False

		return True

	def _penalty_mark_x(self, team: str) -> float:
		if team == "home":
			return self.cfg.pitch_x - self.cfg.penalty_mark_x
		return self.cfg.penalty_mark_x

	def _goal_x(self, team: str) -> float:
		if team == "home":
			return self.cfg.pitch_x
		return 0.0

	def _penalty_box_x_bounds(self, team: str) -> tuple[float, float]:
		if team == "home":
			return self.cfg.pitch_x - self.cfg.penalty_area_x_max, self.cfg.pitch_x
		return 0.0, self.cfg.penalty_area_x_max


# 코너킥 트리거
	def _get_corner_trigger_player(self, row: pd.Series, first_in_row: pd.Series) -> str | None:
		candidates: list[str] = []

		for player in self.players:
			player_x = row[self.x_col(player)]
			player_y = row[self.y_col(player)]
			if pd.isna(player_x) or pd.isna(player_y):
				continue

			corner_x = self.cfg.pitch_x if self.team_of(player) == "home" else 0.0
			near_x = abs(player_x - corner_x) <= self.cfg.corner_tol
			near_top = abs(player_y - self.cfg.pitch_y) <= self.cfg.corner_tol
			near_bottom = abs(player_y - 0.0) <= self.cfg.corner_tol
			if not (near_x and (near_top or near_bottom)):
				continue

			candidates.append(player)

		return self._select_nearest_by_first_in_dist(first_in_row, candidates)


# 스로인 트리거
	def _get_throw_in_trigger_player(self, row: pd.Series, first_in_row: pd.Series) -> str | None:
		candidates: list[str] = []

		for player in self.players:
			player_x = row[self.x_col(player)]
			player_y = row[self.y_col(player)]
			if pd.isna(player_x) or pd.isna(player_y):
				continue

			near_top = player_y > self.cfg.pitch_y - self.cfg.throw_in_tol
			near_bottom = player_y < self.cfg.throw_in_tol
			if not (near_top or near_bottom):
				continue

			candidates.append(player)

		return self._select_nearest_by_first_in_dist(first_in_row, candidates)



# 골킥 트리거
	def _get_goal_kick_trigger_player(self, trigger_row: pd.Series, first_in_row: pd.Series) -> str | None:
		candidates: list[str] = []

		for player in self.players:
			player_x = trigger_row[self.x_col(player)]
			player_y = trigger_row[self.y_col(player)]
			if pd.isna(player_x) or pd.isna(player_y):
				continue

			team = self.team_of(player)
			goal_x = 0.0 if team == "home" else self.cfg.pitch_x
			if goal_x == self.cfg.pitch_x:
				goal_area_x_min = self.cfg.pitch_x - self.cfg.goal_area_x
				goal_area_x_max = self.cfg.pitch_x
			else:
				goal_area_x_min = 0.0
				goal_area_x_max = self.cfg.goal_area_x

			in_x = goal_area_x_min - self.cfg.goal_area_x_tol <= player_x <= goal_area_x_max + self.cfg.goal_area_x_tol
			in_y = abs(player_y - self.cfg.center_y) <= self.cfg.goal_area_y + self.cfg.goal_area_y_tol
			if not (in_x and in_y):
				continue

			candidates.append(player)

		return self._select_nearest_by_first_in_dist(first_in_row, candidates)

	def _select_nearest_by_first_in_dist(self, first_in_row: pd.Series, candidates: list[str]) -> str | None:
		scored_candidates: list[tuple[str, float]] = []
		for player in candidates:
			dist_col = f"dist_{player}"
			if dist_col not in first_in_row.index:
				continue

			dist = first_in_row[dist_col]
			if pd.isna(dist):
				continue

			scored_candidates.append((player, float(dist)))

		if not scored_candidates:
			return None

		scored_candidates.sort(key=lambda item: item[1])
		return scored_candidates[0][0]

	@staticmethod
	def team_of(player_id: str) -> str:
		return _team_of_fn(player_id)

	@staticmethod
	def x_col(player_id: str) -> str:
		return f"{player_id}_x"

	@staticmethod
	def y_col(player_id: str) -> str:
		return f"{player_id}_y"
