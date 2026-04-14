from __future__ import annotations

import pandas as pd

from autoevent.helpers import get_players
from autoevent.setpiece_trigger import SetPieceConfig, SetPieceTrigger
from tools.config import TRACKING_FPS


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
			.add_throw_in_labels()
			.add_goal_kick_labels()
			.add_free_kick_labels()
			._add_fallback_labels()
			.add_deadball_events()
			.tracking
		)

	def _ball_in_player_pz_extended(self, interval: dict, trigger_player: str) -> bool:
		"""dead ball 구간 스캔 + first_in 이후 25프레임 스캔 중 볼이 PZ 안이면 True.

		- dead ball 구간: 선수가 공을 들고 있을 때 볼 위치가 트래킹될 수 있음 (dist≈0)
		- first_in 이후: 킥/던진 직후라도 초반 수 프레임 내에는 여전히 PZ 안일 수 있음
		"""
		dist_col = f"dist_{trigger_player}"
		if dist_col not in self.tracking.columns:
			return False

		r = self.cfg.r_pz
		step = self.cfg.fallback_scan_step
		first_in_idx = interval["first_in_idx"]
		start_idx = interval.get("start_idx")
		end_idx = interval.get("end_idx")

		# 1. dead ball 구간 스캔 (step 간격으로 샘플링)
		if start_idx is not None and end_idx is not None:
			s_pos = self.tracking.index.get_loc(start_idx)
			e_pos = self.tracking.index.get_loc(end_idx)
			for pos in range(s_pos, e_pos + 1, step):
				d = self.tracking.iloc[pos].get(dist_col)
				if pd.notna(d) and float(d) <= r:
					return True

		# 2. first_in 이후 25 프레임 스캔 (alive 구간만)
		fi_pos = self.tracking.index.get_loc(first_in_idx)
		period = self.tracking.at[first_in_idx, "period_id"]
		for pos in range(fi_pos, min(fi_pos + 25, len(self.tracking))):
			row = self.tracking.iloc[pos]
			if row.get("period_id") != period:
				break
			if row.get("ball_state") != "alive":
				break
			d = row.get(dist_col)
			if pd.notna(d) and float(d) <= r:
				return True

		return False

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

		trigger_player = self._get_kickoff_trigger_player(trigger_row, first_in_row)
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

		trigger_player = self._get_penalty_trigger_player(trigger_row, first_in_row)
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

		trigger_player = self._get_corner_trigger_player(trigger_row, first_in_row)
		if trigger_player is None:
			return False

		if not self._ball_in_player_pz(first_in_row, trigger_player):
			return False

		interval["trigger_player"] = trigger_player
		return True

	def add_throw_in_labels(self) -> 'SetPieceDetector':
		for interval in self.intervals:
			if not self._is_throw_in_interval(interval):
				continue

			trigger_player = interval.get("trigger_player")
			if pd.isna(trigger_player):
				continue

			first_in_idx = interval["first_in_idx"]
			if pd.notna(self.tracking.at[first_in_idx, "set_piece_type"]):
				continue

			self.tracking.at[first_in_idx, "trigger_player"] = trigger_player
			self.tracking.at[first_in_idx, "trigger_team"] = self.team_of(trigger_player)
			self.tracking.at[first_in_idx, "set_piece_type"] = "throw_in"

		return self

	def _is_throw_in_interval(self, interval: dict[str, object]) -> bool:
		end_idx = interval["end_idx"]
		first_in_idx = interval["first_in_idx"]

		first_in_row = self.tracking.loc[first_in_idx]
		trigger_row = first_in_row if end_idx is None else self.tracking.loc[end_idx]
		if trigger_row["period_id"] != first_in_row["period_id"]:
			trigger_row = first_in_row

		trigger_player = self._get_throw_in_trigger_player(trigger_row, first_in_row)
		if trigger_player is None:
			return False

		if not self._ball_in_player_pz(first_in_row, trigger_player):
			return False

		interval["trigger_player"] = trigger_player
		return True

	def add_goal_kick_labels(self) -> 'SetPieceDetector':
		for interval in self.intervals:
			if not self._is_goal_kick_interval(interval):
				continue

			trigger_player = interval.get("trigger_player")
			if pd.isna(trigger_player):
				continue

			first_in_idx = interval["first_in_idx"]
			if pd.notna(self.tracking.at[first_in_idx, "set_piece_type"]):
				continue

			self.tracking.at[first_in_idx, "trigger_player"] = trigger_player
			self.tracking.at[first_in_idx, "trigger_team"] = self.team_of(trigger_player)
			self.tracking.at[first_in_idx, "set_piece_type"] = "goal_kick"

		return self

	def _is_goal_kick_interval(self, interval: dict[str, object]) -> bool:
		end_idx = interval["end_idx"]
		first_in_idx = interval["first_in_idx"]

		first_in_row = self.tracking.loc[first_in_idx]
		trigger_row = first_in_row if end_idx is None else self.tracking.loc[end_idx]
		if trigger_row["period_id"] != first_in_row["period_id"]:
			trigger_row = first_in_row

		trigger_player = self._get_goal_kick_trigger_player(trigger_row, first_in_row)
		if trigger_player is None:
			return False

		if not self._ball_in_player_pz(first_in_row, trigger_player):
			return False

		interval["trigger_player"] = trigger_player
		return True

	def add_free_kick_labels(self) -> 'SetPieceDetector':
		for interval in self.intervals:
			if not self._is_free_kick_interval(interval):
				continue

			trigger_player = interval.get("trigger_player")
			if pd.isna(trigger_player):
				continue

			first_in_idx = interval["first_in_idx"]
			if pd.notna(self.tracking.at[first_in_idx, "set_piece_type"]):
				continue

			self.tracking.at[first_in_idx, "trigger_player"] = trigger_player
			self.tracking.at[first_in_idx, "trigger_team"] = self.team_of(trigger_player)
			self.tracking.at[first_in_idx, "set_piece_type"] = "free_kick"

		return self

	def _is_free_kick_interval(self, interval: dict[str, object]) -> bool:
		first_in_idx = interval["first_in_idx"]
		first_in_row = self.tracking.loc[first_in_idx]

		if pd.notna(self.tracking.at[first_in_idx, "set_piece_type"]):
			return False

		scored_candidates: list[tuple[str, float]] = []
		for player in self.players:
			dist_col = f"dist_{player}"
			if dist_col not in first_in_row.index:
				continue

			dist = first_in_row[dist_col]
			if pd.isna(dist):
				continue

			scored_candidates.append((player, float(dist)))

		if not scored_candidates:
			return False

		scored_candidates.sort(key=lambda item: item[1])
		trigger_player = scored_candidates[0][0]
		if not self._ball_in_player_pz(first_in_row, trigger_player):
			return False

		interval["trigger_player"] = trigger_player
		return True

	def add_deadball_events(self) -> 'SetPieceDetector':
		self.tracking["deadball_event"] = pd.NA
		period_start_frames = set(
			self.tracking.index[
				self.tracking["period_id"].ne(self.tracking["period_id"].shift(1))
			]
		)

		event_by_setpiece = {
			"kickoff": "goal",
			"penalty_kick": "penalty awarded",
			"corner_kick": "out for corner kick",
			"throw_in": "out for throw-in",
			"goal_kick": "out for goalkick",
			"free_kick": "foul",
		}

		for interval in self.intervals:
			first_in_idx = interval["first_in_idx"]
			set_piece_type = self.tracking.at[first_in_idx, "set_piece_type"]
			if pd.isna(set_piece_type):
				continue

			# 전/후반 시작 kickoff는 deadball_event(=goal)로 기록하지 않음
			if str(set_piece_type) == "kickoff" and first_in_idx in period_start_frames:
				continue

			event_name = event_by_setpiece.get(str(set_piece_type))
			if event_name is None:
				continue

			start_idx = interval.get("start_idx")
			event_idx = None

			if pd.notna(start_idx):
				start_pos = self.tracking.index.get_loc(start_idx)
				if start_pos > 0:
					prev_idx = self.tracking.index[start_pos - 1]
					event_idx = prev_idx

					# 전/후반 시작 kickoff는 deadball_event를 기록하지 않음
					if str(set_piece_type) == "kickoff":
						prev_period = self.tracking.at[prev_idx, "period_id"]
						start_period = self.tracking.at[start_idx, "period_id"]
						if prev_period != start_period:
							event_idx = None
				else:
					if str(set_piece_type) == "kickoff":
						event_idx = None
					else:
						event_idx = start_idx
			else:
				if str(set_piece_type) == "kickoff":
					event_idx = None

			if event_idx is None:
				continue

			self.tracking.at[event_idx, "deadball_event"] = event_name

		return self

	# ── Fallback: Issues 2 & 3 from paper Supplementary S1 ─────────────────

	def _add_fallback_labels(self) -> 'SetPieceDetector':
		"""Phase 1에서 미탐지된 dead ball 구간을 Issue 2, 3으로 재시도.

		Issue 2 (인바운딩 선수 트래킹 소실):
		  dead ball 구간 내에서 corner/throw-in 트리거가 활성화됐다가
		  선수가 트래킹에서 사라진 경우 → 해당 타입으로 할당.

		Issue 3 (first_in 프레임에서 볼이 PZ 밖):
		  PZ(패턴) 조건 없이 트리거만 확인하는 계층적 탐색:
		  K → P → C → T → Ci* → Ti* → G → F?
		"""
		for interval in self.intervals:
			first_in_idx = interval["first_in_idx"]
			if pd.notna(self.tracking.at[first_in_idx, "set_piece_type"]):
				continue  # Phase 1에서 이미 할당됨
			if self._try_ballz_throw_in(interval):
				continue
			if self._try_issue2_assign(interval):
				continue
			self._try_issue3_assign(interval)
		return self

	def _try_ballz_throw_in(self, interval: dict) -> bool:
		"""ball_z 기반 throw_in 판정:
		1. first_in_row 자체에 is_loss가 있으면 그 ball_z 확인.
		2. 없으면 window 내에서 controller_id == trigger_player이고 is_loss인
		   가장 가까운 행을 탐색해 그 ball_z 확인.
		ball_z >= THROW_IN_BALL_Z_MIN이면 throw_in 할당.
		"""
		if "ball_z" not in self.tracking.columns:
			return False
		if "is_loss" not in self.tracking.columns:
			return False

		first_in_idx = interval["first_in_idx"]
		start_idx = interval.get("start_idx")
		first_in_row = self.tracking.loc[first_in_idx]

		# throw_in trigger player 탐색 (first_in_row 기준)
		player = self._get_throw_in_trigger_player(first_in_row, first_in_row)
		if player is None:
			return False

		# 1. first_in_row 자체에 is_loss가 있으면 우선 사용
		if self.tracking.at[first_in_idx, "is_loss"] == True:
			ball_z = self.tracking.at[first_in_idx, "ball_z"]
			if pd.notna(ball_z) and float(ball_z) >= self.cfg.throw_in_ball_z_min:
				self._assign_fallback(first_in_idx, "throw_in", player)
				return True
			return False

		# 2. window 내에서 controller_id == trigger_player이고 is_loss인 가장 가까운 행 탐색
		ref_pos = (
			self.tracking.index.get_loc(start_idx)
			if start_idx is not None
			else self.tracking.index.get_loc(first_in_idx)
		)
		window_frames = int(self.cfg.sp_loss_map_window * TRACKING_FPS)
		search_start = max(0, ref_pos - window_frames)

		loss_idx = None
		for j in range(ref_pos - 1, search_start - 1, -1):
			jidx = self.tracking.index[j]
			ctrl = self.tracking.at[jidx, "controller_id"]
			if (self.tracking.at[jidx, "is_loss"] == True
					and pd.notna(ctrl) and ctrl == player):
				loss_idx = jidx
				break

		if loss_idx is None:
			return False

		ball_z = self.tracking.at[loss_idx, "ball_z"]
		if pd.isna(ball_z) or float(ball_z) < self.cfg.throw_in_ball_z_min:
			return False

		self._assign_fallback(first_in_idx, "throw_in", player)
		return True

	def _assign_fallback(self, first_in_idx, sp_type: str, player: str) -> None:
		self.tracking.at[first_in_idx, "trigger_player"] = player
		self.tracking.at[first_in_idx, "trigger_team"] = self.team_of(player)
		self.tracking.at[first_in_idx, "set_piece_type"] = sp_type

	def _scan_for_incomplete_ct(
		self, interval: dict
	) -> tuple[str | None, str | None]:
		"""Dead ball 구간을 스캔하여 불완전(incomplete) corner(C*)/throw-in(T*) 트리거 탐색.

		불완전 트리거 조건:
		  - di 프레임에서 트리거가 활성화됐으나 di < dc (dc = end_idx)이고,
		  - [di+1, dc] 구간에서 해당 선수가 절반 이상 NaN (out-of-bounds 소실)
		    OR first_in_frame에서 NaN.

		Returns (player_id, sp_type) or (None, None).
		"""
		start_idx = interval.get("start_idx")
		end_idx = interval.get("end_idx")
		first_in_idx = interval["first_in_idx"]
		if start_idx is None or end_idx is None:
			return None, None

		start_pos = self.tracking.index.get_loc(start_idx)
		end_pos = self.tracking.index.get_loc(end_idx)
		if start_pos >= end_pos:
			return None, None

		first_in_row = self.tracking.loc[first_in_idx]
		step = self.cfg.fallback_scan_step

		last_c: dict[str, int] = {}  # player → last iloc where corner trigger active
		last_t: dict[str, int] = {}  # player → last iloc where throw-in trigger active

		for pos in range(start_pos, end_pos + 1, step):
			row = self.tracking.iloc[pos]
			c_player = self._get_corner_trigger_player(row, first_in_row)
			if c_player is not None:
				last_c[c_player] = pos
			t_player = self._get_throw_in_trigger_player(row, first_in_row)
			if t_player is not None:
				last_t[t_player] = pos

		def player_absent_after(player: str, after_pos: int) -> bool:
			total = missing = 0
			for pos in range(after_pos + step, end_pos + 1, step):
				if pd.isna(self.tracking.iloc[pos][self.x_col(player)]):
					missing += 1
				total += 1
			return total > 0 and missing / total >= 0.5

		def is_incomplete(player: str, last_pos: int) -> bool:
			return (
				last_pos < end_pos and player_absent_after(player, last_pos)
			) or pd.isna(self.tracking.at[first_in_idx, self.x_col(player)])

		best_c: str | None = None
		best_c_pos = -1
		for player, pos in last_c.items():
			if is_incomplete(player, pos) and pos > best_c_pos:
				best_c_pos = pos
				best_c = player

		best_t: str | None = None
		best_t_pos = -1
		for player, pos in last_t.items():
			if is_incomplete(player, pos) and pos > best_t_pos:
				best_t_pos = pos
				best_t = player

		if best_c is not None and best_t is not None:
			return (best_c, "corner_kick") if best_c_pos >= best_t_pos else (best_t, "throw_in")
		if best_c is not None:
			return best_c, "corner_kick"
		if best_t is not None:
			return best_t, "throw_in"
		return None, None

	def _try_issue2_assign(self, interval: dict) -> bool:
		"""Issue 2: 인바운딩 선수가 경계선 밖으로 나가 트래킹 소실된 경우."""
		player, sp_type = self._scan_for_incomplete_ct(interval)
		if player is None:
			return False
		first_in_idx = interval["first_in_idx"]
		interval["trigger_player"] = player
		self._assign_fallback(first_in_idx, sp_type, player)
		return True

	def _try_issue3_assign(self, interval: dict) -> bool:
		"""Issue 3: first_in_frame에서 볼이 PZ 밖 → trigger-only 계층 탐색.

		K → P → C(complete) → T(complete) → Ci* → Ti* → G → F?
		PZ(패턴) 조건 없이 트리거만 확인.
		"""
		end_idx = interval.get("end_idx")
		first_in_idx = interval["first_in_idx"]
		if end_idx is None:
			return False

		end_row = self.tracking.loc[end_idx]

		# K: 킥오프 트리거
		player = self._get_kickoff_trigger_player(end_row, end_row)
		if player is not None:
			interval["trigger_player"] = player
			self._assign_fallback(first_in_idx, "kickoff", player)
			return True

		# P: 페널티킥 트리거 + setup 확인
		player = self._get_penalty_trigger_player(end_row, end_row)
		if player is not None and self._is_penalty_setup(end_row, player):
			interval["trigger_player"] = player
			self._assign_fallback(first_in_idx, "penalty_kick", player)
			return True

		# C complete: end_idx에서 코너킥 트리거 활성
		player = self._get_corner_trigger_player(end_row, end_row)
		if player is not None:
			interval["trigger_player"] = player
			self._assign_fallback(first_in_idx, "corner_kick", player)
			return True

		# T complete: end_idx에서 스로인 트리거 활성
		player = self._get_throw_in_trigger_player(end_row, end_row)
		if player is not None:
			interval["trigger_player"] = player
			self._assign_fallback(first_in_idx, "throw_in", player)
			return True

		# Ci*/Ti*: incomplete 트리거 (선수 소실)
		i_player, i_sp_type = self._scan_for_incomplete_ct(interval)
		if i_player is not None:
			interval["trigger_player"] = i_player
			self._assign_fallback(first_in_idx, i_sp_type, i_player)
			return True

		# G: 골킥 트리거
		player = self._get_goal_kick_trigger_player(end_row, end_row)
		if player is not None:
			interval["trigger_player"] = player
			self._assign_fallback(first_in_idx, "goal_kick", player)
			return True

		# F?: 어떤 트리거도 없음 → end_row 기준 가장 가까운 선수로 free_kick 할당
		scored: list[tuple[float, str]] = []
		for p in self.players:
			dist_col = f"dist_{p}"
			if dist_col not in end_row.index:
				continue
			d = end_row[dist_col]
			if not pd.isna(d):
				scored.append((float(d), p))
		if scored:
			scored.sort()
			player = scored[0][1]
			interval["trigger_player"] = player
			self._assign_fallback(first_in_idx, "free_kick", player)
			return True

		return False


