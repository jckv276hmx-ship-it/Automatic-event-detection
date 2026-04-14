from __future__ import annotations

import numpy as np
import pandas as pd

from autoevent.helpers import (
    get_players,
    team_of,
    player_xy,
    in_shot_zone,
    in_cross_zone,
    in_attacking_pa,
    detect_gks,
    attacker_in_pa,
    ball_toward_goal,
    shot_on_target,
)
from tools.config import TRACKING_FPS, SP_LOSS_MAP_WINDOW


class OpenPlayEventDetector:
    """Classifies in-game events (pass, cross, shot, reception, interception, save, ...).

    Input must be the output of SetPieceDetector.run() which already contains:
        is_gain, is_loss, gain_player, loss_player,
        control_sequence_id, controller_team,
        set_piece_type, deadball_event, deadball_id,
        ball_dir_out_x, ball_dir_out_y, ball_x, ball_y
    """

    def __init__(self, tracking: pd.DataFrame):
        self.tracking = tracking.copy()
        self.players = get_players(self.tracking)
        self.gk_ids = detect_gks(self.tracking, self.players)

    def run(self) -> pd.DataFrame:
        self.tracking["event_name"] = pd.NA
        self.tracking["event_player"] = pd.NA
        self.tracking["event_team"] = pd.NA

        self._classify_kicking_events()
        self._classify_gain_events()

        return self.tracking

    def _classify_kicking_events(self) -> None:
        tracking = self.tracking
        for lid in self.valid_loss_idx():
            loss_row = tracking.loc[lid]
            player = loss_row.get("loss_player")
            if pd.isna(player):
                continue
            team = team_of(str(player))
            loss_iloc = tracking.index.get_loc(lid)
            shooter_event, side_events = self._is_shot(loss_iloc, team)
            event = shooter_event if shooter_event is not None else "pass"
            tracking.at[lid, "event_name"] = event
            tracking.at[lid, "event_player"] = player
            tracking.at[lid, "event_team"] = team
            # 세이빙 이벤트 등 부가 이벤트 기록
            for se in side_events:
                tracking.at[se["idx"], "event_name"] = se["event_name"]
                tracking.at[se["idx"], "event_player"] = se["player"]
                tracking.at[se["idx"], "event_team"] = se["team"]

    def _classify_gain_events(self) -> None:
        """is_gain 행에서 직전 이벤트(event_name 있는 행)와 팀 비교 후
        같은 팀 → reception, 다른 팀 → interception 을 기록."""
        tracking = self.tracking
        gain_rows = tracking.index[tracking["is_gain"].fillna(False) == True]

        for gid in gain_rows:
            gp = tracking.at[gid, "gain_player"]
            if pd.isna(gp):
                continue
            gain_team = team_of(str(gp))
            gain_iloc = tracking.index.get_loc(gid)

            # 직전 event_name 이 있는 행을 역방향 탐색
            prev_event_team = None
            for j in range(gain_iloc - 1, -1, -1):
                ev = tracking.iloc[j].get("event_name")
                if pd.notna(ev):
                    prev_event_team = tracking.iloc[j].get("event_team")
                    break

            if prev_event_team is None:
                continue

            if gain_team == str(prev_event_team):
                event = "reception"
            else:
                event = "interception"

            tracking.at[gid, "event_name"] = event
            tracking.at[gid, "event_player"] = gp
            tracking.at[gid, "event_team"] = gain_team

    def _is_shot(self, loss_iloc: int, team: str) -> tuple[str | None, list[dict]]:
        """kick 이후 시퀀스를 스캔해 이벤트 유형과 부가 이벤트(GK save 등)를 반환.

        Rule 1 – shooter → out for goalkick (gain 없음):
            shot zone  → shot_off_target
            cross zone → cross
            other      → pass

        Rule 2 – shooter → GK gain → out for goalkick (데이터 오류 케이스):
            shot zone  → shot_off_target
            cross zone → cross
            other      → pass

        Rule 3 – shooter → (GK gain →) out for corner kick:
            cross zone → cross  (GK save_deflect if GK gain)
            other/shot zone → shot_on/off_target  (GK save_deflect if GK gain)

        Returns (shooter_event | None, side_events)
        """
        tracking = self.tracking
        loss_row = tracking.iloc[loss_iloc]
        bx = float(loss_row.get("ball_x", np.nan))
        by = float(loss_row.get("ball_y", np.nan))

        gk_gain_seen = False
        gk_gain_idx = None
        gk_gain_player = None
        first_gain_idx = None
        first_gain_player = None
        first_gain_iloc = None

        for i in range(loss_iloc + 1, len(tracking)):
            row = tracking.iloc[i]
            idx = tracking.index[i]

            if row.get("ball_state") != "alive":
                break

            if row.get("is_gain") == True:
                gp = row.get("gain_player")
                if not gk_gain_seen and pd.notna(gp) and str(gp) in self.gk_ids and team_of(str(gp)) != team:
                    gk_gain_seen = True
                    gk_gain_idx = idx
                    gk_gain_player = str(gp)
                    continue
                # Rule 6 용: break 전에 첫 번째 non-GK gain 저장
                if first_gain_idx is None:
                    first_gain_idx = idx
                    first_gain_player = str(gp) if pd.notna(gp) else None
                    first_gain_iloc = i
                break

            db_ev = row.get("deadball_event")
            if pd.notna(db_ev):
                db_ev = str(db_ev)
                # Rule 1 & 2
                if db_ev == "out for goalkick":
                    if in_shot_zone(bx, by, team):
                        return "shot_off_target", []
                    elif in_cross_zone(bx, by, team):
                        return "cross", []
                    else:
                        return "pass", []
                # Rule 3
                if db_ev == "out for corner kick":
                    side_events = []
                    if gk_gain_idx is not None:
                        side_events.append({
                            "idx": gk_gain_idx,
                            "event_name": "save_deflect",
                            "player": gk_gain_player,
                            "team": team_of(gk_gain_player),
                        })
                    if in_cross_zone(bx, by, team):
                        return "cross", side_events
                    else:
                        on_target = self._detect_shot_on_target(loss_iloc, team)
                        shooter_ev = "shot_on_target" if on_target else "shot_off_target"
                        return shooter_ev, side_events
                # Rule 4
                if db_ev == "goal":
                    side_events = []
                    if gk_gain_idx is not None:
                        side_events.append({
                            "idx": gk_gain_idx,
                            "event_name": "unsuccessful_save",
                            "player": gk_gain_player,
                            "team": team_of(gk_gain_player),
                        })
                    return "shot_on_target", side_events
                return None, []

        # Rule 5 – shooter → GK gain (deadball 없이 끝남)
        if gk_gain_seen and gk_gain_idx is not None:
            gk_gain_iloc = self.tracking.index.get_loc(gk_gain_idx)
            gk_gain_row = self.tracking.iloc[gk_gain_iloc]
            gk_gain_team = team_of(gk_gain_player)
            if in_shot_zone(bx, by, team):
                if ball_toward_goal(loss_row, team):
                    on_target = self._detect_shot_on_target(loss_iloc, team)
                    shooter_ev = "shot_on_target" if on_target else "shot_off_target"
                    save_ev = self._gk_save_type(gk_gain_iloc, gk_gain_player)
                    return shooter_ev, [{
                        "idx": gk_gain_idx,
                        "event_name": save_ev,
                        "player": gk_gain_player,
                        "team": gk_gain_team,
                    }]
                else:
                    return "pass", [{
                        "idx": gk_gain_idx,
                        "event_name": "reception_from_loose_ball",
                        "player": gk_gain_player,
                        "team": gk_gain_team,
                    }]
            elif in_cross_zone(bx, by, team):
                if attacker_in_pa(gk_gain_row, team, self.players, self.gk_ids):
                    save_ev = self._gk_save_type(gk_gain_iloc, gk_gain_player)
                    return "cross", [{
                        "idx": gk_gain_idx,
                        "event_name": save_ev,
                        "player": gk_gain_player,
                        "team": gk_gain_team,
                    }]
                else:
                    return "pass", [{
                        "idx": gk_gain_idx,
                        "event_name": "reception_from_loose_ball",
                        "player": gk_gain_player,
                        "team": gk_gain_team,
                    }]
            else:
                return "pass", [{
                    "idx": gk_gain_idx,
                    "event_name": "reception_from_loose_ball",
                    "player": gk_gain_player,
                    "team": gk_gain_team,
                }]

        # Rule 6 – cross zone kick, non-GK gain 선수가 attacking PA 안
        if first_gain_idx is not None and first_gain_player is not None:
            if in_cross_zone(bx, by, team):
                first_gain_row = tracking.iloc[first_gain_iloc]
                gp_px, gp_py = player_xy(first_gain_row, first_gain_player)
                if in_attacking_pa(gp_px, gp_py, team) and attacker_in_pa(first_gain_row, team, self.players, self.gk_ids):
                    return "cross", []

        return None, []

    def _find_shot_save_sequences(self, valid_loss_idx: list) -> list[dict]:
        """shot-save sequence 탐색.

        shooter의 loss 프레임에서 출발해 순방향으로 스캔:
          - ball_state != "alive" → dead 직전 alive 프레임을 end_idx로 종료
          - 상대팀 GK의 gain → 허용 (저장 후 계속 스캔)
          - GK gain 이전 다른 선수 gain → shot 아님, 즉시 중단
          - GK gain 이후 다른 선수 gain → GK gain까지로 종료 (gk_gain_cutoff)

        전제: loss 프레임의 볼이 shot_zone 안이고 ball_toward_goal이어야 함.
        """
        tracking = self.tracking
        sequences = []

        for lid in valid_loss_idx:
            loss_player = tracking.at[lid, "loss_player"]
            if pd.isna(loss_player):
                continue
            team = team_of(str(loss_player))

            loss_iloc = tracking.index.get_loc(lid)
            loss_row = tracking.iloc[loss_iloc]
            bx = float(loss_row.get("ball_x", np.nan))
            by = float(loss_row.get("ball_y", np.nan))

            # 전제: shot zone + ball toward goal
            if not in_shot_zone(bx, by, team):
                continue
            if not ball_toward_goal(loss_row, team):
                continue

            gk_gain_idx = None
            gk_gain_player = None
            seq_end_idx = None
            seq_end_type = None
            prev_idx = None

            for i in range(loss_iloc + 1, len(tracking)):
                row = tracking.iloc[i]
                idx = tracking.index[i]

                if row.get("ball_state") != "alive":
                    seq_end_idx = prev_idx if prev_idx is not None else idx
                    seq_end_type = "deadball"
                    break

                prev_idx = idx

                if row.get("is_gain") == True:
                    gp = row.get("gain_player")
                    if pd.isna(gp):
                        continue
                    gp = str(gp)
                    gp_team = gp.split("_")[0]

                    if gk_gain_idx is None:
                        if gp in self.gk_ids and gp_team != team:
                            gk_gain_idx = idx
                            gk_gain_player = gp
                        else:
                            break
                    else:
                        seq_end_idx = gk_gain_idx
                        seq_end_type = "gk_gain_cutoff"
                        break

            if seq_end_idx is not None:
                sequences.append({
                    "loss_idx":    lid,
                    "end_idx":     seq_end_idx,
                    "end_type":    seq_end_type,
                    "gk_gain_idx": gk_gain_idx,
                    "gk_player":   gk_gain_player,
                })

        return sequences

    def valid_loss_idx(self):
        tracking = self.tracking
        gain_idx = tracking.index[tracking["is_gain"].fillna(False)]
        loss_idx = tracking.index[tracking["is_loss"].fillna(False)]

        seen: set = set()
        valid_loss_idx = []

        # 조건 1: gain_player == loss_player (open play)
        for gid in gain_idx:
            pos = int(loss_idx.searchsorted(gid, side="right"))
            if pos < len(loss_idx):
                lid = loss_idx[pos]
                if lid in seen:
                    continue
                gp = tracking.at[gid, "gain_player"]
                lp = tracking.at[lid, "loss_player"]
                if pd.notna(gp) and pd.notna(lp) and gp == lp:
                    seen.add(lid)
                    valid_loss_idx.append(lid)

        # 조건 2: set_piece_type 수행자의 바로 다음 loss (set piece kick)
        sp_rows = tracking[tracking["set_piece_type"].notna()]
        for sp_idx in sp_rows.index:
            sp_player = tracking.at[sp_idx, "controller_id"]
            if pd.isna(sp_player):
                continue
            pos = int(loss_idx.searchsorted(sp_idx, side="left"))
            if pos < len(loss_idx):
                lid = loss_idx[pos]
                if lid in seen:
                    continue
                lp = tracking.at[lid, "loss_player"]
                if pd.notna(lp) and lp == sp_player:
                    seen.add(lid)
                    valid_loss_idx.append(lid)

        # 조건 3: set piece 직전 window 내 미매핑 loss → set piece 원인 kick 분류용
        # (볼이 곧장 라인 밖으로 나가서 gain이 없는 경우 조건 1에서 누락됨)
        sp_loss_window = int(SP_LOSS_MAP_WINDOW * TRACKING_FPS)
        for sp_idx in sp_rows.index:
            sp_pos = tracking.index.get_loc(sp_idx)
            window_start = max(0, sp_pos - sp_loss_window)
            for j in range(sp_pos - 1, window_start - 1, -1):
                jidx = tracking.index[j]
                if jidx in seen:
                    continue
                if tracking.at[jidx, "is_loss"] == True:
                    seen.add(jidx)
                    valid_loss_idx.append(jidx)
                    break

        return valid_loss_idx
        

    # ------------------------------------------------------------------
    def _detect_shot_on_target(self, loss_iloc: int, team: str) -> bool:
        """loss 이후 GK gain 또는 deadball_event 중 먼저 나오는 프레임으로 on target 판정."""
        tracking = self.tracking
        prev_row = None

        for i in range(loss_iloc + 1, len(tracking)):
            row = tracking.iloc[i]

            if row.get("ball_state") != "alive":
                if prev_row is not None:
                    return shot_on_target(prev_row, team)
                return False

            if row.get("deadball_event") is not None and pd.notna(row.get("deadball_event")):
                return shot_on_target(row, team)

            if row.get("is_gain") == True:
                gp = row.get("gain_player")
                if pd.notna(gp) and str(gp) in self.gk_ids and team_of(str(gp)) != team:
                    return shot_on_target(row, team)
                break

            prev_row = row

        return False

    # ------------------------------------------------------------------ #
    def _gk_save_type(self, gk_gain_iloc: int, gk_player: str) -> str:
        """GK gain 이후 1초(TRACKING_FPS 프레임) 내에 소유를 잃으면 save_deflect,
        아니면 save_retain을 반환."""
        tracking = self.tracking
        window = TRACKING_FPS
        end = min(gk_gain_iloc + window + 1, len(tracking))

        for i in range(gk_gain_iloc + 1, end):
            row = tracking.iloc[i]
            if row.get("is_loss") == True and str(row.get("loss_player")) == str(gk_player):
                return "save_deflect"
            if row.get("ball_state") != "alive":
                break

        return "save_retain"

