from __future__ import annotations

import pandas as pd

from autoevent.poss import get_players
from autoevent.setpiece_trigger import SetPieceConfig, SetPieceTrigger


class SetPieceDetector(SetPieceTrigger):

	def __init__(self, tracking: pd.DataFrame, cfg: SetPieceConfig | None = None):
		self.tracking = tracking.copy()
		self.cfg = cfg or SetPieceConfig()
		self.players = get_players(self.tracking)
		self.intervals: list[dict[str, object]] = []

	def run(self) -> pd.DataFrame:
		return (
			self.add_dead_ball_intervals()
			.add_first_in_frames()
			.add_kickoff_labels()
			.add_penalty_labels()
			.add_corner_labels()
			.tracking
		)

	def add_dead_ball_intervals(self) -> 'SetPieceDetector':
		self.tracking["deadball_id"] = pd.NA
		self.intervals = []

		sequence_id = 0
		in_deadball = False
		start_idx = None

		for idx, is_dead in self.tracking["ball_state"].eq("dead").items():
			if is_dead and not in_deadball:
				sequence_id += 1
				in_deadball = True
				start_idx = idx

			if is_dead:
				self.tracking.at[idx, "deadball_id"] = sequence_id

			if not is_dead and in_deadball:
				end_idx = self.tracking.index[self.tracking.index.get_loc(idx) - 1]
				self.intervals.append(
					{
						"deadball_id": sequence_id,
						"start_idx": start_idx,
						"end_idx": end_idx,
						"first_in_idx": idx,
					}
				)
				in_deadball = False
				start_idx = None

		# 경기 시작 시 첫 번째 인 플레이 프레임이 속한 구간이 존재하지 않는 경우를 위해, 각 기간의 첫 번째 인 플레이 프레임을 구간으로 추가
		period_starts = self.tracking["period_id"].ne(self.tracking["period_id"].shift(1))
		existing_first_in = {interval["first_in_idx"] for interval in self.intervals}
		for idx in self.tracking.index[period_starts]:
			if idx in existing_first_in:
				continue

			if self.tracking.at[idx, "ball_state"] == "dead":
				continue

			self.intervals.append(
				{
					"deadball_id": pd.NA,
					"start_idx": None,
					"end_idx": None,
					"first_in_idx": idx,
				}
			)

		return self

	def add_first_in_frames(self) -> 'SetPieceDetector':
		self.tracking["first_in_frame"] = False

		for interval in self.intervals:
			first_in_idx = interval["first_in_idx"]
			self.tracking.at[first_in_idx, "first_in_frame"] = True

		return self

	def add_kickoff_labels(self) -> 'SetPieceDetector':
		self.tracking["trigger_player"] = pd.NA
		self.tracking["trigger_team"] = pd.NA
		self.tracking["set_piece_type"] = pd.NA

		for interval in self.intervals:
			if not self._is_kickoff_interval(interval):
				continue

			trigger_player = interval.get("trigger_player")
			if pd.isna(trigger_player):
				continue

			first_in_idx = interval["first_in_idx"]
			self.tracking.at[first_in_idx, "trigger_player"] = trigger_player
			self.tracking.at[first_in_idx, "trigger_team"] = self.team_of(trigger_player)
			self.tracking.at[first_in_idx, "set_piece_type"] = "kickoff"

		return self

	def _is_kickoff_interval(self, interval: dict[str, object]) -> bool:
		end_idx = interval["end_idx"]
		first_in_idx = interval["first_in_idx"]

		first_in_row = self.tracking.loc[first_in_idx]
		# 킥오프 트리거 판단 시, 첫 번째 인 플레이 프레임이 속한 구간의 마지막 프레임을 트리거 판단에 활용
		trigger_row = first_in_row if end_idx is None else self.tracking.loc[end_idx]
		if trigger_row["period_id"] != first_in_row["period_id"]:
			trigger_row = first_in_row

		trigger_player = self._get_kickoff_trigger_player(trigger_row)
		if trigger_player is None:
			return False

		if not self._ball_in_player_pz(first_in_row, trigger_player):
			return False

		interval["trigger_player"] = trigger_player
		return True
	
	def add_penalty_labels(self) -> 'SetPieceDetector':
		for interval in self.intervals:
			if not self._is_penalty_interval(interval):
				continue

			trigger_player = interval.get("trigger_player")
			if pd.isna(trigger_player):
				continue

			first_in_idx = interval["first_in_idx"]
			if pd.notna(self.tracking.at[first_in_idx, "set_piece_type"]):
				continue

			self.tracking.at[first_in_idx, "trigger_player"] = trigger_player
			self.tracking.at[first_in_idx, "trigger_team"] = self.team_of(trigger_player)
			self.tracking.at[first_in_idx, "set_piece_type"] = "penalty_kick"

		return self

	def _is_penalty_interval(self, interval: dict[str, object]) -> bool:
		end_idx = interval["end_idx"]
		first_in_idx = interval["first_in_idx"]

		first_in_row = self.tracking.loc[first_in_idx]
		trigger_row = first_in_row if end_idx is None else self.tracking.loc[end_idx]
		if trigger_row["period_id"] != first_in_row["period_id"]:
			trigger_row = first_in_row

		trigger_player = self._get_penalty_trigger_player(trigger_row)
		if trigger_player is None:
			return False

		if not self._is_penalty_setup(trigger_row, trigger_player):
			return False

		if not self._ball_in_player_pz(first_in_row, trigger_player):
			return False

		interval["trigger_player"] = trigger_player
		return True

	def add_corner_labels(self) -> 'SetPieceDetector':
		for interval in self.intervals:
			if not self._is_corner_interval(interval):
				continue

			trigger_player = interval.get("trigger_player")
			if pd.isna(trigger_player):
				continue

			first_in_idx = interval["first_in_idx"]
			if pd.notna(self.tracking.at[first_in_idx, "set_piece_type"]):
				continue

			self.tracking.at[first_in_idx, "trigger_player"] = trigger_player
			self.tracking.at[first_in_idx, "trigger_team"] = self.team_of(trigger_player)
			self.tracking.at[first_in_idx, "set_piece_type"] = "corner_kick"

		return self

	def _is_corner_interval(self, interval: dict[str, object]) -> bool:
		end_idx = interval["end_idx"]
		first_in_idx = interval["first_in_idx"]

		first_in_row = self.tracking.loc[first_in_idx]
		trigger_row = first_in_row if end_idx is None else self.tracking.loc[end_idx]
		if trigger_row["period_id"] != first_in_row["period_id"]:
			trigger_row = first_in_row

		trigger_player = self._get_corner_trigger_player(trigger_row)
		if trigger_player is None:
			return False

		if not self._ball_in_player_pz(first_in_row, trigger_player):
			return False

		interval["trigger_player"] = trigger_player
		return True



