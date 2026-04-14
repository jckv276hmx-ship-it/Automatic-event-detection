import os
import sys
from datetime import timedelta
from typing import Dict, Optional

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation, axes, collections, lines, text

import tools.matplotsoccer as mps
from tools import config

anim_config = {
    "figsize": (9, 6),
    "fontsize": 15,
    "player_size": 400,
    "ball_size": 150,
    "player_history": 20,
    "ball_history": 50,
}


class Animator:
    def __init__(
        self,
        track_dict: Dict[str, pd.DataFrame],
        events: Optional[pd.DataFrame] = None,
        show_times: bool = True,
        show_events: bool = False,
        play_speed: int = 1,
    ):
        self.track_dict = track_dict
        self.show_times = show_times
        self.show_events = show_events
        self.play_speed = play_speed
        self.arg_dict = dict()

        if events is None or events.empty:
            self.events = None
        else:
            if "frame_id" in track_dict["main"].columns:
                valid_frame_ids = track_dict["main"]["frame_id"].unique()
                self.events = events[events["frame_id"].isin(valid_frame_ids)]
            else:
                self.events = events

    @staticmethod
    def plot_players(tracking: pd.DataFrame, ax: axes.Axes, alpha=1):
        if len(tracking.columns) == 0:
            return None

        color = "tab:red" if tracking.columns[0].startswith("home_") else "tab:blue"
        x = tracking[tracking.columns[0::2]].values
        y = tracking[tracking.columns[1::2]].values
        scat = ax.scatter(x[0], y[0], s=anim_config["player_size"], c=color, alpha=alpha, zorder=2)

        players = [c[:-2] for c in tracking.columns[0::2]]
        plots = dict()
        annots = dict()

        for p in players:
            (plots[p],) = ax.plot([], [], c=color, alpha=alpha, ls=":", zorder=0)
            annots[p] = ax.annotate(
                int(p.split("_")[-1]),
                xy=tracking.loc[0, [f"{p}_x", f"{p}_y"]],
                ha="center",
                va="center",
                color="w",
                fontsize=anim_config["fontsize"] - 2,
                fontweight="bold",
                annotation_clip=False,
                zorder=3,
            )
            annots[p].set_animated(True)

        return tracking, scat, plots, annots

    @staticmethod
    def animate_players(
        t: int,
        inplay_records: pd.DataFrame,
        tracking: pd.DataFrame,
        scat: collections.PatchCollection,
        plots: Dict[str, lines.Line2D],
        annots: Dict[str, text.Annotation],
    ):
        x = tracking[tracking.columns[0::2]].values
        y = tracking[tracking.columns[1::2]].values
        scat.set_offsets(np.stack([x[t], y[t]]).T)

        for p in plots.keys():
            inplay_start = inplay_records.at[p, "start_index"]
            inplay_end = inplay_records.at[p, "end_index"]

            if t >= inplay_start:
                if t <= inplay_end:
                    t_from = max(t - anim_config["player_history"] + 1, inplay_start)
                    plots[p].set_data(tracking.loc[t_from:t, f"{p}_x"], tracking.loc[t_from:t, f"{p}_y"])
                    annots[p].set_position(tracking.loc[t, [f"{p}_x", f"{p}_y"]].values)
                elif t == inplay_end + 1:
                    plots[p].set_alpha(0)
                    annots[p].set_alpha(0)

    @staticmethod
    def plot_ball(xy: pd.DataFrame, ax=axes.Axes, color="w", edgecolor="k", marker="o", show_path=True):
        x = xy.values[:, 0]
        y = xy.values[:, 1]
        scat = ax.scatter(
            x,
            y,
            s=anim_config["ball_size"],
            c=color,
            edgecolors=edgecolor,
            marker=marker,
            zorder=4,
        )

        if show_path:
            pathcolor = "k" if color == "w" else color
            (plot,) = ax.plot([], [], pathcolor, zorder=3)
        else:
            plot = None

        return x, y, scat, plot

    @staticmethod
    def animate_ball(
        t: int,
        x: np.ndarray,
        y: np.ndarray,
        scat: collections.PatchCollection,
        plot: lines.Line2D = None,
    ):
        scat.set_offsets(np.array([x[t], y[t]]))

        if plot is not None:
            t_from = max(t - anim_config["ball_history"], 0)
            plot.set_data(x[t_from : t + 1], y[t_from : t + 1])

    @staticmethod
    def _short_player_label(player: object) -> str:
        if pd.isna(player):
            return ""
        player_str = str(player)
        if not player_str:
            return ""

        tokens = player_str.split("|")
        short_tokens = []
        for token in tokens:
            tok = token.strip()
            if tok.startswith("home_"):
                short_tokens.append("h_" + tok[5:])
            elif tok.startswith("away_"):
                short_tokens.append("a_" + tok[5:])
            else:
                short_tokens.append(tok)
        return "|".join(short_tokens)

    @staticmethod
    def _normalize_event_type_label(event_type: object) -> str:
        if pd.isna(event_type):
            return ""
        label = str(event_type).strip()
        if label in {"possession_loss", "loss"}:
            return "loss"
        if label in {"possession_gain", "gain"}:
            return "gain"
        return label

    @staticmethod
    def _compose_setpiece_label(set_piece_type: object, deadball_event: object) -> str:
        sp = "" if pd.isna(set_piece_type) else str(set_piece_type).strip()
        db = "" if pd.isna(deadball_event) else str(deadball_event).strip()
        if sp and db:
            return f"{sp} | {db}"
        return sp or db

    @staticmethod
    def _build_table_events(events: Optional[pd.DataFrame], main_tracking: pd.DataFrame) -> Optional[pd.DataFrame]:
        base_events = None
        if events is not None and not events.empty:
            base_events = events.copy()
            if "control" not in base_events.columns:
                if "event_type" in base_events.columns:
                    base_events["control"] = base_events["event_type"]
                elif "possession" in base_events.columns:
                    base_events["control"] = base_events["possession"]
                else:
                    base_events["control"] = ""

            if "setpiece" not in base_events.columns:
                if "set_piece_type" in base_events.columns or "deadball_event" in base_events.columns:
                    base_events["setpiece"] = base_events.apply(
                        lambda row: Animator._compose_setpiece_label(
                            row.get("set_piece_type"), row.get("deadball_event")
                        ),
                        axis=1,
                    )
                else:
                    base_events["setpiece"] = ""

        extra_parts = []

        # ── gain 이벤트 ───────────────────────────────────────────────
        if "is_gain" in main_tracking.columns:
            gain_rows = main_tracking[main_tracking["is_gain"] == True].copy()
            if not gain_rows.empty:
                gain_player = gain_rows.get(
                    "gain_player", pd.Series([""] * len(gain_rows), index=gain_rows.index)
                )
                tg = pd.DataFrame({
                    "timestamp": gain_rows["timestamp"],
                    "control": "gain",
                    "player_id": gain_player,
                    "setpiece": "",
                })
                if "frame_id" in gain_rows.columns:
                    tg["frame_id"] = gain_rows["frame_id"]
                if "period_id" in gain_rows.columns:
                    tg["period_id"] = gain_rows["period_id"]
                extra_parts.append(tg)

        # ── loss 이벤트 ───────────────────────────────────────────────
        if "is_loss" in main_tracking.columns:
            loss_rows = main_tracking[main_tracking["is_loss"] == True].copy()
            if not loss_rows.empty:
                loss_player = loss_rows.get(
                    "loss_player", pd.Series([""] * len(loss_rows), index=loss_rows.index)
                )
                tl = pd.DataFrame({
                    "timestamp": loss_rows["timestamp"],
                    "control": "loss",
                    "player_id": loss_player,
                    "setpiece": "",
                })
                if "frame_id" in loss_rows.columns:
                    tl["frame_id"] = loss_rows["frame_id"]
                if "period_id" in loss_rows.columns:
                    tl["period_id"] = loss_rows["period_id"]
                extra_parts.append(tl)

        # ── duel 이벤트 (연속 구간 첫 프레임만) ──────────────────────
        if "ball_control" in main_tracking.columns and "duel_players" in main_tracking.columns:
            duel_rows = main_tracking[main_tracking["ball_control"].eq("duel")].copy()
            if not duel_rows.empty:
                keep_mask = duel_rows["duel_players"] != duel_rows["duel_players"].shift(1)
                duel_rows = duel_rows[keep_mask]
                td = pd.DataFrame({
                    "timestamp": duel_rows["timestamp"],
                    "control": "duel",
                    "player_id": duel_rows["duel_players"],
                    "setpiece": "",
                })
                if "frame_id" in duel_rows.columns:
                    td["frame_id"] = duel_rows["frame_id"]
                if "period_id" in duel_rows.columns:
                    td["period_id"] = duel_rows["period_id"]
                extra_parts.append(td)

        table_setpiece = pd.DataFrame()
        if "set_piece_type" in main_tracking.columns or "deadball_event" in main_tracking.columns:
            setpiece_rows = main_tracking.copy()
            sp_series = setpiece_rows.get("set_piece_type", pd.Series([pd.NA] * len(setpiece_rows), index=setpiece_rows.index))
            db_series = setpiece_rows.get("deadball_event", pd.Series([pd.NA] * len(setpiece_rows), index=setpiece_rows.index))
            setpiece_mask = sp_series.notna() | db_series.notna()
            setpiece_rows = setpiece_rows.loc[setpiece_mask]

            if not setpiece_rows.empty:
                sp_labels = setpiece_rows.apply(
                    lambda row: Animator._compose_setpiece_label(
                        row.get("set_piece_type"), row.get("deadball_event")
                    ),
                    axis=1,
                )
                # 연속된 같은 setpiece 구간에서 첫 프레임만 유지 (flood 방지)
                keep_mask = sp_labels != sp_labels.shift(1)
                setpiece_rows = setpiece_rows[keep_mask]
                sp_labels = sp_labels[keep_mask]

                table_setpiece = pd.DataFrame(
                    {
                        "timestamp": setpiece_rows["timestamp"],
                        "control": "setpiece",
                        "player_id": setpiece_rows.get("trigger_player", pd.Series([""] * len(setpiece_rows), index=setpiece_rows.index)),
                        "setpiece": sp_labels,
                    }
                )
                if "frame_id" in setpiece_rows.columns:
                    table_setpiece["frame_id"] = setpiece_rows["frame_id"]
                if "period_id" in setpiece_rows.columns:
                    table_setpiece["period_id"] = setpiece_rows["period_id"]

        if not table_setpiece.empty:
            extra_parts.append(table_setpiece)

        if base_events is None:
            if not extra_parts:
                return None
            parts = pd.concat(extra_parts, ignore_index=True, sort=False)
        else:
            concat_frames = [base_events] + extra_parts
            parts = pd.concat(concat_frames, ignore_index=True, sort=False)
            dedup_cols = [c for c in ["frame_id", "control", "player_id", "setpiece"] if c in parts.columns]
            if dedup_cols:
                parts = parts.drop_duplicates(subset=dedup_cols, keep="first")

        # 같은 frame의 gain/loss/duel을 open-play 이벤트 행에 control_tag로 병합 (중복 행 제거)
        parts["control_tag"] = ""
        CTRL_EVTS = {"gain", "loss", "duel"}
        group_key = [c for c in ["period_id", "frame_id"] if c in parts.columns]
        if not group_key:
            group_key = [c for c in ["period_id", "timestamp"] if c in parts.columns]
        rows_to_drop = []
        for _, grp in parts.groupby(group_key, sort=False):
            ctrl_mask = grp["control"].isin(CTRL_EVTS)
            open_mask = ~ctrl_mask & ~grp["control"].eq("setpiece")
            if ctrl_mask.any() and open_mask.any():
                ctrl_label = grp.loc[ctrl_mask, "control"].iloc[0]
                for idx in grp[open_mask].index:
                    parts.at[idx, "control_tag"] = ctrl_label
                rows_to_drop.extend(grp[ctrl_mask].index.tolist())
        if rows_to_drop:
            parts = parts.drop(rows_to_drop).reset_index(drop=True)

        # possession 정보 추가: frame_id 또는 nearest timestamp로 controller_team 조회
        if "controller_team" in main_tracking.columns:
            if "frame_id" in parts.columns and "frame_id" in main_tracking.columns:
                poss_map = main_tracking.set_index("frame_id")["controller_team"]
                parts["possession"] = parts["frame_id"].map(poss_map).fillna("").astype(str)
                parts["possession"] = parts["possession"].replace("<NA>", "").replace("nan", "")
            elif "timestamp" in parts.columns:
                ts_map = main_tracking.set_index("timestamp")["controller_team"]
                parts["possession"] = parts["timestamp"].map(ts_map).fillna("").astype(str)
                parts["possession"] = parts["possession"].replace("<NA>", "").replace("nan", "")
        return parts

    @staticmethod
    def plot_event_table(events: pd.DataFrame, fig: plt.Figure, table_ax: plt.Axes, frame_times: np.ndarray) -> dict:
        if events is None or table_ax is None or frame_times is None or events.empty:
            return None

        assert "timestamp" in events.columns
        events = events.copy()
        # 누적초 계산: period(half) 컬럼이 있으면 후반은 45*60초 더함
        if isinstance(events["timestamp"].iloc[0], timedelta):
            events["_ts"] = events["timestamp"].dt.total_seconds()
        else:
            events["_ts"] = pd.to_numeric(events["timestamp"], errors="coerce")
        if "period_id" in events.columns:
            # period==2(후반)이면 45*60초 더함
            events["_ts_total"] = events["_ts"] + (events["period_id"] == 2) * 45 * 60
        else:
            events["_ts_total"] = events["_ts"]

        events = events.dropna(subset=["_ts_total"]).sort_values("_ts_total").reset_index(drop=True)
        event_times = events["_ts_total"].to_numpy()
        raw_controls = events.get("control")
        if raw_controls is None:
            raw_controls = events.get("event_type")
        if raw_controls is None and "possession" in events.columns:
            raw_controls = events["possession"]
        if raw_controls is None:
            raw_controls = pd.Series([""] * len(events), dtype=object)
        controls = raw_controls.apply(Animator._normalize_event_type_label).to_numpy(dtype=str)

        raw_setpieces = events.get("setpiece")
        if raw_setpieces is None:
            if "set_piece_type" in events.columns or "deadball_event" in events.columns:
                raw_setpieces = events.apply(
                    lambda row: Animator._compose_setpiece_label(
                        row.get("set_piece_type"), row.get("deadball_event")
                    ),
                    axis=1,
                )
            else:
                raw_setpieces = pd.Series([""] * len(events), dtype=object)
        setpieces = raw_setpieces.fillna("").astype(str).to_numpy(dtype=str)

        raw_possessions = events.get("possession")
        if raw_possessions is None:
            raw_possessions = pd.Series([""] * len(events), dtype=object)
        possessions = raw_possessions.fillna("").astype(str).replace("<NA>", "").replace("nan", "").to_numpy(dtype=str)

        raw_control_tags = events.get("control_tag")
        if raw_control_tags is None:
            raw_control_tags = pd.Series([""] * len(events), dtype=object)
        control_tags = raw_control_tags.fillna("").astype(str).to_numpy(dtype=str)

        event_players = events.get("player_id", pd.Series([""] * len(events), dtype=object)).copy()
        if "duel_players" in events.columns:
            duel_mask = pd.Series(controls).astype(str).eq("duel")
            event_players.loc[duel_mask] = events.loc[duel_mask, "duel_players"]
        event_players = event_players.apply(Animator._short_player_label).to_numpy(dtype=str)

        FONT = anim_config["fontsize"] - 3
        # 고정 분수 좌표 사용 (픽셀 계산 불안정 회피)
        LINE_H  = 0.042   # 한 줄 높이 (axes 분수 좌표)
        BLOCK_H = 0.078   # 이벤트 1개 블록 높이
        HEADER_H = LINE_H * 1.5
        max_events = max(1, int((1.0 - HEADER_H) / BLOCK_H))

        # 헤더
        table_ax.text(0.02, 1.0 - LINE_H * 0.1, "time   player  event",
                      fontsize=FONT, fontfamily="monospace", ha="left", va="top",
                      fontweight="bold")
        table_ax.plot([0, 1], [1.0 - HEADER_H, 1.0 - HEADER_H],
                      transform=table_ax.transAxes, color="black", linewidth=0.8, clip_on=False)

        # 이벤트 행: 이벤트 1개당 텍스트 객체 2개 (줄1: time+player, 줄2: control)
        event_row1 = []
        event_row2 = []
        start_y = 1.0 - HEADER_H - LINE_H * 0.4
        for i in range(max_events):
            y1 = start_y - i * BLOCK_H
            y2 = y1 - LINE_H * 1.05
            r1 = table_ax.text(0.02, y1, "", fontsize=FONT, fontfamily="monospace", ha="left", va="top")
            r2 = table_ax.text(0.02, y2, "", fontsize=FONT - 1, fontfamily="monospace", ha="left", va="top",
                               color="dimgray")
            event_row1.append(r1)
            event_row2.append(r2)

        return {
            "event_row1": event_row1,
            "event_row2": event_row2,
            "event_times": event_times,
            "event_players": event_players,
            "controls": controls,
            "setpieces": setpieces,
            "possessions": possessions,
            "control_tags": control_tags,
            "event_queue": [],
            "event_ptr": 0,
        }

    @staticmethod
    def animate_event_table(t: int, frame_times: np.ndarray, event_attr: dict):
        if event_attr is None or frame_times is None:
            return

        event_row1    = event_attr["event_row1"]
        event_row2    = event_attr["event_row2"]
        event_times   = event_attr["event_times"]
        event_players = event_attr["event_players"]
        controls      = event_attr["controls"]
        setpieces     = event_attr["setpieces"]
        possessions   = event_attr.get("possessions", [])
        control_tags  = event_attr.get("control_tags", [])
        event_queue   = event_attr["event_queue"]  # list of (line1, line2) tuples
        event_ptr     = event_attr["event_ptr"]

        current_time = frame_times[t]
        if np.isfinite(current_time):
            while event_ptr < len(event_times) and event_times[event_ptr] <= current_time + 1e-9:
                sec = float(event_times[event_ptr])
                mm = int(sec // 60)
                ss = sec % 60
                time_str   = f"{mm:02d}:{ss:05.2f}"
                node_id     = event_players[event_ptr] if event_ptr < len(event_players) else ""
                control     = controls[event_ptr]      if event_ptr < len(controls)      else ""
                setpiece    = setpieces[event_ptr]     if event_ptr < len(setpieces)     else ""
                control_tag = control_tags[event_ptr]  if event_ptr < len(control_tags)  else ""
                # 제어 타입에 따라 line1/line2 구성
                CONTROL_EVENTS = {"gain", "loss", "duel"}
                if control in CONTROL_EVENTS:
                    # 독립 gain/loss/duel: line1은 time+player만, line2에 타입
                    line1 = f"{time_str}  {node_id}"
                    line2 = f"  {control}"
                elif control == "setpiece" or (not control and setpiece):
                    # setpiece: line1에 setpiece label, line2는 비움
                    sp_label = setpiece if setpiece else "setpiece"
                    line1 = f"{time_str}  {node_id}  {sp_label}"
                    line2 = ""
                else:
                    # open play 이벤트: line1에 event, line2에 control_tag (gain/loss 등이 있으면)
                    line1 = f"{time_str}  {node_id}  {control}".rstrip()
                    line2 = f"  {control_tag}" if control_tag else ""
                event_queue.append((line1, line2))
                if len(event_queue) > len(event_row1):
                    event_queue.pop(0)
                event_ptr += 1

            for i in range(len(event_row1)):
                if i < len(event_queue):
                    event_row1[i].set_text(event_queue[i][0])
                    event_row2[i].set_text(event_queue[i][1])
                else:
                    event_row1[i].set_text("")
                    event_row2[i].set_text("")

        event_attr["event_ptr"] = event_ptr

    def plot_init(self, ax: axes.Axes):
        tracking = self.track_dict["main"].iloc[:: self.play_speed].copy()
        tracking = tracking.dropna(axis=1, how="all").reset_index(drop=True)
        xy_cols = [c for c in tracking.columns if c.endswith("_x") or c.endswith("_y")]

        inplay_records = []
        for c in xy_cols[::2]:
            inplay_index = tracking[tracking[c].notna()].index
            inplay_records.append([c[:-2], inplay_index[0], inplay_index[-1]])
        inplay_records = pd.DataFrame(inplay_records, columns=["object", "start_index", "end_index"])

        home_tracking = tracking[[c for c in xy_cols if c.startswith("home_")]].fillna(-100)
        away_tracking = tracking[[c for c in xy_cols if c.startswith("away_")]].fillna(-100)

        home_state = self.plot_players(home_tracking, ax, alpha=1)
        away_state = self.plot_players(away_tracking, ax, alpha=1)

        ball_state = None
        if "ball_x" in tracking.columns and tracking["ball_x"].notna().any():
            ball_xy = tracking[["ball_x", "ball_y"]]
            ball_state = Animator.plot_ball(ball_xy, ax, "w", "k", "o")

        self.track_dict["main"] = tracking
        self.arg_dict["main"] = {
            "inplay_records": inplay_records.set_index("object"),
            "home": home_state,
            "away": away_state,
            "ball": ball_state,
        }

    def run(self, max_frames=np.inf, fps=10):
        # ── Figure 설정 ──────────────────────────────────────────────────────
        if self.show_events and self.events is not None:
            fig = plt.figure(figsize=(anim_config["figsize"][0] * 1.35, anim_config["figsize"][1]))
            gs = fig.add_gridspec(1, 2, width_ratios=[4.5, 1.5], wspace=0.05)
            ax = fig.add_subplot(gs[0, 0])
            table_ax = fig.add_subplot(gs[0, 1])
            table_ax.axis("off")
            table_ax.set_xlim(0, 1)
            table_ax.set_ylim(0, 1)
        else:
            fig, ax = plt.subplots(figsize=anim_config["figsize"])
            table_ax = None

        mps.field("green", config.PITCH_X, config.PITCH_Y, fig, ax, show=False)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_axis_off()
        fig.subplots_adjust(left=0.02, right=0.92, bottom=0.05, top=0.95)

        self.plot_init(ax)

        main_tracking = self.track_dict["main"]
        text_y = config.PITCH_Y + 1

        # ── 타임스탬프 ────────────────────────────────────────────────────────
        frame_times = None
        if "timestamp" in main_tracking.columns:
            ts = main_tracking["timestamp"]
            if isinstance(ts.iloc[0], timedelta):
                frame_times = ts.dt.total_seconds().to_numpy()
            else:
                frame_times = pd.to_numeric(ts, errors="coerce").to_numpy()

        if self.show_times and frame_times is not None:
            timestamps_str = np.array([f"{int(s // 60):02d}:{s % 60:05.2f}" for s in frame_times])
            time_text = ax.text(0, text_y, timestamps_str[0],
                                fontsize=anim_config["fontsize"], ha="left", va="bottom")
            time_text.set_animated(True)

        # ── 이벤트 테이블 ─────────────────────────────────────────────────────
        event_attr = None
        if self.show_events:
            table_events = Animator._build_table_events(self.events, main_tracking)
            if table_events is not None:
                event_attr = Animator.plot_event_table(table_events, fig, table_ax, frame_times)

        # ── animate 함수 ──────────────────────────────────────────────────────
        inplay_records = self.arg_dict["main"]["inplay_records"]
        home_state     = self.arg_dict["main"]["home"]
        away_state     = self.arg_dict["main"]["away"]
        ball_state     = self.arg_dict["main"]["ball"]

        def animate(t):
            if home_state is not None:
                Animator.animate_players(t, inplay_records, *home_state)
            if away_state is not None:
                Animator.animate_players(t, inplay_records, *away_state)
            if ball_state is not None:
                Animator.animate_ball(t, *ball_state)

            if self.show_times and frame_times is not None:
                time_text.set_text(timestamps_str[t])

            if self.show_events and event_attr is not None:
                Animator.animate_event_table(t, frame_times, event_attr)

        frames = min(max_frames, main_tracking.shape[0])
        anim = animation.FuncAnimation(fig, animate, frames=int(frames), interval=1000 / fps)
        plt.close(fig)
        return anim
