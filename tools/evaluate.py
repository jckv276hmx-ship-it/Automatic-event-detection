"""
Evaluation utilities for Automatic Event Detection.

Provides:
  - Label maps (GT_TYPE_MAP, SP_MAP_GT, PRED_TYPE_MAP)
  - evaluate()        : time-window greedy matching (set piece)
  - evaluate_paper()  : time-window + player greedy matching (open play)
  - aggregate()       : compute P/R/F1 from per-match rows
  - prepare_gt()      : build gt_eval / gt_sp from a SportecData object
  - prepare_pred()    : build pred_full / pred_sp from pipeline outputs
  - prepare_gt_goals(): build goal GT DataFrame from a SportecData object
  - prepare_pred_goals(): build goal pred DataFrame from pipeline output
  - evaluate_goals()  : time-window greedy matching for goal detection
  - eval_match()      : run full evaluation for one match
  - eval_all_matches(): run eval_match() across a list of match IDs
"""

from __future__ import annotations

import pandas as pd

from tools.match_data import MatchData
from tools.sportec_data import SportecData

# ── Label maps ────────────────────────────────────────────────────────────────

GT_TYPE_MAP: dict[str, str] = {
    "Pass":         "pass",
    "Cross":        "cross",
    "Shot":         "shot",
    "Interception": "interception",
}

SP_MAP_GT: dict[str, str] = {
    "ThrowIn":    "throw_in",
    "GoalKick":   "goal_kick",
    "CornerKick": "corner_kick",
    "FreeKick":   "free_kick",
    "KickOff":    "kickoff",
    "Penalty":    "penalty_kick",
}

PRED_TYPE_MAP: dict[str, str] = {
    "pass":           "pass",
    "cross":          "cross",
    "shot_on_target": "shot",
    "shot_off_target":"shot",
    "interception":   "interception",
}

OPEN_LABELS: list[str] = ["pass", "cross", "shot", "interception"]
SP_LABELS:   list[str] = ["throw_in", "goal_kick", "corner_kick", "free_kick", "kickoff", "penalty_kick"]

# Default evaluation windows (seconds)
PAPER_WINDOW: float = 10.0   # open play: ±10s + player
SP_WINDOW:    float = 10.0   # set piece: ±10s


# ── Core evaluation functions ─────────────────────────────────────────────────

def evaluate(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    label: str,
    window: float = SP_WINDOW,
) -> dict:
    """
    Time-window greedy 1:1 matching.

    Both DataFrames must have columns: period_id, timestamp, label.
    """
    gt_sub   = gt_df[gt_df["label"] == label][["period_id", "timestamp"]].copy().reset_index(drop=True)
    pred_sub = pred_df[pred_df["label"] == label][["period_id", "timestamp"]].copy().reset_index(drop=True)

    matched_gt   = set()
    matched_pred = set()

    for gi, gr in gt_sub.iterrows():
        candidates = pred_sub[
            (pred_sub["period_id"] == gr["period_id"]) &
            (abs(pred_sub["timestamp"] - gr["timestamp"]) <= window)
        ]
        candidates = candidates[~candidates.index.isin(matched_pred)]
        if candidates.empty:
            continue
        best_pi = (candidates["timestamp"] - gr["timestamp"]).abs().idxmin()
        matched_gt.add(gi)
        matched_pred.add(best_pi)

    tp = len(matched_gt)
    fp = len(pred_sub) - len(matched_pred)
    fn = len(gt_sub) - tp
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    return {
        "label": label, "GT": len(gt_sub), "Pred": len(pred_sub),
        "TP": tp, "FP": fp, "FN": fn,
        "Precision": round(p, 3), "Recall": round(r, 3), "F1": round(f1, 3),
    }


def evaluate_paper(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    label: str,
    window: float = PAPER_WINDOW,
    use_player: bool = True,
) -> dict:
    """
    Time-window greedy 1:1 matching with optional player constraint.

    gt_df   must have: period_id, timestamp, object_id, label
    pred_df must have: period_id, timestamp, event_player, label
    """
    gt_sub   = gt_df[gt_df["label"] == label][["period_id", "timestamp", "object_id"]].copy().reset_index(drop=True)
    pred_sub = pred_df[pred_df["label"] == label][["period_id", "timestamp", "event_player"]].copy().reset_index(drop=True)

    matched_gt   = set()
    matched_pred = set()

    for gi, gr in gt_sub.iterrows():
        cand = pred_sub[
            (pred_sub["period_id"] == gr["period_id"]) &
            (abs(pred_sub["timestamp"] - gr["timestamp"]) <= window)
        ]
        if use_player:
            cand = cand[cand["event_player"] == gr["object_id"]]
        cand = cand[~cand.index.isin(matched_pred)]
        if cand.empty:
            continue
        best_pi = (cand["timestamp"] - gr["timestamp"]).abs().idxmin()
        matched_gt.add(gi)
        matched_pred.add(best_pi)

    tp = len(matched_gt)
    fp = len(pred_sub) - len(matched_pred)
    fn = len(gt_sub) - tp
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    return {
        "label": label, "GT": len(gt_sub), "Pred": len(pred_sub),
        "TP": tp, "FP": fp, "FN": fn,
        "Precision": round(p, 3), "Recall": round(r, 3), "F1": round(f1, 3),
    }


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate(rows: list[dict]) -> pd.DataFrame:
    """
    Aggregate per-match (or per-label) result rows into a summary DataFrame
    with micro-averaged P/R/F1 columns.
    """
    df = pd.DataFrame(rows)
    summary = df.groupby("label")[["GT", "Pred", "TP", "FP", "FN"]].sum()
    summary["Precision"] = (summary["TP"] / (summary["TP"] + summary["FP"])).round(3)
    summary["Recall"]    = (summary["TP"] / (summary["TP"] + summary["FN"])).round(3)
    denom = summary["Precision"] + summary["Recall"]
    summary["F1"] = (2 * summary["Precision"] * summary["Recall"] / denom.where(denom > 0, other=1)).round(3)
    return summary


def micro_summary(summary_df: pd.DataFrame) -> dict:
    """Return micro-averaged totals from an aggregate() result."""
    tp = summary_df["TP"].sum()
    fp = summary_df["FP"].sum()
    fn = summary_df["FN"].sum()
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"TP": tp, "FP": fp, "FN": fn,
            "Precision": round(p, 3), "Recall": round(r, 3), "F1": round(f1, 3)}


# ── Data preparation helpers ──────────────────────────────────────────────────

def prepare_gt(match: SportecData) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build GT DataFrames from a SportecData object.

    Returns
    -------
    gt_eval : open-play GT  (columns: period_id, timestamp, object_id, label)
    gt_sp   : set-piece GT  (columns: period_id, timestamp, object_id, label)
    """
    gt_raw = SportecData.find_object_ids(match.lineup, match.events)
    gt_raw = SportecData.find_spadl_event_types(gt_raw)
    gt = MatchData.calculate_event_seconds(gt_raw)

    gt_eval = gt[gt["event_type"].isin(GT_TYPE_MAP)].copy()
    gt_eval["label"] = gt_eval["event_type"].map(GT_TYPE_MAP)

    gt_sp = gt[gt["set_piece_type"].isin(SP_MAP_GT)].copy()
    gt_sp["label"] = gt_sp["set_piece_type"].map(SP_MAP_GT)

    return gt_eval, gt_sp


def prepare_gt_goals(match: SportecData) -> pd.DataFrame:
    """
    Build goal GT DataFrame from a SportecData object.

    Returns
    -------
    gt_goals : columns: period_id, timestamp
    """
    from tools.match_data import MatchData
    gt_raw = SportecData.find_object_ids(match.lineup, match.events)
    gt_raw = SportecData.find_spadl_event_types(gt_raw)
    gt_all = MatchData.calculate_event_seconds(gt_raw)
    gt_goals = gt_all[
        (gt_all["event_type"] == "Shot") & (gt_all["result"] == "Goal")
    ][["period_id", "timestamp"]].copy().reset_index(drop=True)
    gt_goals["label"] = "goal"
    return gt_goals


def prepare_pred_goals(result: pd.DataFrame) -> pd.DataFrame:
    """
    Build goal pred DataFrame from pipeline output.

    Parameters
    ----------
    result : output of run_pipeline()  (has deadball_event column)

    Returns
    -------
    pred_goals : columns: period_id, timestamp, label
    """
    pred_goals = result[result["deadball_event"] == "goal"][["period_id", "timestamp"]].copy().reset_index(drop=True)
    pred_goals["label"] = "goal"
    return pred_goals


def evaluate_goals(
    gt_goals: pd.DataFrame,
    pred_goals: pd.DataFrame,
    window: float = SP_WINDOW,
) -> dict:
    """
    Time-window greedy 1:1 matching for goal detection.
    Wraps evaluate() with label='goal'.
    """
    return evaluate(gt_goals, pred_goals, label="goal", window=window)


def prepare_pred(result: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build prediction DataFrames from pipeline outputs.

    Parameters
    ----------
    result : output of run_pipeline()           (has set_piece_type column)
    test   : output of OpenPlayEventDetector.run() (has event_name, event_player columns)

    Returns
    -------
    pred_full : open-play pred  (columns: period_id, timestamp, event_player, label)
    pred_sp   : set-piece pred  (columns: period_id, timestamp, label)
    """
    pred_full = test[test["event_name"].isin(PRED_TYPE_MAP)].copy()
    pred_full["label"] = pred_full["event_name"].map(PRED_TYPE_MAP)

    pred_sp = result[result["set_piece_type"].notna()][
        ["period_id", "timestamp", "set_piece_type"]
    ].copy()
    pred_sp = pred_sp.rename(columns={"set_piece_type": "label"}).reset_index(drop=True)

    return pred_full, pred_sp


# ── Single-match evaluation ───────────────────────────────────────────────────

def eval_match(
    match: SportecData,
    result: pd.DataFrame,
    test: pd.DataFrame,
    open_window: float = PAPER_WINDOW,
    sp_window: float = SP_WINDOW,
    use_player: bool = True,
) -> tuple[list[dict], list[dict], dict]:
    """
    Run open-play, set-piece, and goal evaluation for a single match.

    Returns
    -------
    open_rows  : list of result dicts (one per open-play label)
    sp_rows    : list of result dicts (one per set-piece label)
    goal_row   : single result dict for goal detection
    """
    gt_eval, gt_sp = prepare_gt(match)
    pred_full, pred_sp = prepare_pred(result, test)
    gt_goals = prepare_gt_goals(match)
    pred_goals = prepare_pred_goals(result)

    open_rows = [evaluate_paper(gt_eval, pred_full, lbl, window=open_window, use_player=use_player)
                 for lbl in OPEN_LABELS]
    sp_rows   = [evaluate(gt_sp, pred_sp, lbl, window=sp_window)
                 for lbl in SP_LABELS]
    goal_row  = evaluate_goals(gt_goals, pred_goals)

    return open_rows, sp_rows, goal_row


# ── All-matches evaluation ────────────────────────────────────────────────────

def eval_all_matches(
    match_ids: list[str],
    run_pipeline_fn,
    open_play_detector_cls,
    open_window: float = PAPER_WINDOW,
    sp_window: float = SP_WINDOW,
    use_player: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Evaluate all matches and return aggregated open-play, set-piece, and goal results.

    Parameters
    ----------
    match_ids          : list of match ID strings
    run_pipeline_fn    : callable(tracking) → result DataFrame
    open_play_detector_cls : class with .run() method (e.g. OpenPlayEventDetector)
    open_window        : time tolerance for open-play matching (seconds)
    sp_window          : time tolerance for set-piece / goal matching (seconds)
    use_player         : whether to require player match for open-play TP

    Returns
    -------
    sum_open : aggregated open-play DataFrame  (indexed by label)
    sum_sp   : aggregated set-piece DataFrame  (indexed by label)
    sum_goal : aggregated goal DataFrame       (indexed by label)
    """
    all_open, all_sp, all_goal = [], [], []

    for mid in match_ids:
        if verbose:
            print(f"Processing {mid}...", end=" ", flush=True)

        m = SportecData(mid)
        r = run_pipeline_fn(m.tracking)
        t = open_play_detector_cls(r).run()

        open_rows, sp_rows, goal_row = eval_match(m, r, t, open_window, sp_window, use_player)

        for row in open_rows:
            row["match"] = mid
            all_open.append(row)
        for row in sp_rows:
            row["match"] = mid
            all_sp.append(row)
        goal_row["match"] = mid
        all_goal.append(goal_row)

        if verbose:
            print("done")

    return aggregate(all_open), aggregate(all_sp), aggregate(all_goal)
