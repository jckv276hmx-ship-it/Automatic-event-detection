from __future__ import annotations

import pandas as pd

from autoevent.poss import PossessionDetector
from autoevent.set import SetPieceDetector
from autoevent.setpiece_trigger import SetPieceConfig
from autoevent.open import OpenPlayEventDetector


OPEN_PLAY_REQUIRED_COLUMNS = [
	"period_id",
	"frame_id",
	"timestamp",
	"ball_state",
	"ball_x",
	"ball_y",
	"ball_dx_prev",
	"ball_dy_prev",
	"ball_dx_next",
	"ball_dy_next",
	"ball_speed_prev",
	"ball_speed_next",
	"ball_dir_in_x",
	"ball_dir_in_y",
	"ball_dir_out_x",
	"ball_dir_out_y",
	"ball_displacement",
	"ball_control",
	"controller_id",
	"controller_team",
	"controller_distance",
	"duel_players",
	"control_sequence_id",
	"control_sequence_type",
	"control_sequence_player",
	"is_loss",
	"loss_player",
	"loss_team",
	"is_gain",
	"gain_player",
	"gain_team",
	"deadball_id",
	"first_in_frame",
	"trigger_player",
	"trigger_team",
	"set_piece_type",
	"deadball_event",
	"event_name",
	"event_player",
	"event_team",
]


def _select_open_play_columns(df: pd.DataFrame) -> pd.DataFrame:
	player_cols = [
		col for col in df.columns
		if (col.startswith("home_") or col.startswith("away_"))
		and (col.endswith("_x") or col.endswith("_y"))
	]
	available = [col for col in OPEN_PLAY_REQUIRED_COLUMNS if col in df.columns]
	return df[available + player_cols].copy()


def run_pipeline(
	tracking: pd.DataFrame,
	setpiece_cfg: SetPieceConfig | None = None,
) -> pd.DataFrame:
	"""Run possession/set-piece detection and return only open-play required columns."""
	poss_result = PossessionDetector(tracking).run()
	setpiece_result = SetPieceDetector(poss_result, cfg=setpiece_cfg).run()
	return _select_open_play_columns(setpiece_result)
