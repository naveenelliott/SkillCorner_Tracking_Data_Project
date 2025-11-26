import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import bisect
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# CONFIG
ROOT = Path('..')
DATA_DIR = ROOT / "Our Datasets"
EVENTS_CSV = DATA_DIR / "processed_recoverys_and_interceptions_dynamic_events.csv"
TRACKING_CSV = DATA_DIR / "processed_tracking_data_start.csv"
OUT_DIR = ROOT / "Our Datasets"
OUT_DIR.mkdir(exist_ok=True)
OUT_FILE = OUT_DIR / "features_snapshot.parquet"


def parse_match_id_from_unique(unique_id: str) -> Optional[int]:
    if pd.isna(unique_id):
        return None
    try:
        return int(str(unique_id).split('_', 1)[0])
    except Exception:
        return None


def group_tracking_by_match_and_frame(tracking_df: pd.DataFrame) -> Dict[int, List[int]]:
    mapping = {}
    for mid, g in tracking_df.groupby("match_id"):
        mapping[int(mid)] = sorted(g["frame"].unique().tolist())
    return mapping


def build_frames_dict(tracking_df: pd.DataFrame, match_id: int) -> Dict[int, pd.DataFrame]:
    dfm = tracking_df[tracking_df["match_id"] == match_id]
    groups = {int(frame): grp.reset_index(drop=True) for frame, grp in dfm.groupby("frame")}
    return groups


def nearest_available_frame(target_frame: int, available_frames: List[int]) -> int:
    if not available_frames:
        raise ValueError("No available frames to choose from")
    pos = bisect.bisect_left(available_frames, target_frame)
    if pos == 0:
        return available_frames[0]
    if pos >= len(available_frames):
        return available_frames[-1]
    before = available_frames[pos - 1]
    after = available_frames[pos]
    return before if (target_frame - before) <= (after - target_frame) else after


# Core spatial computations
def compute_snapshot_features_for_event(
    event_row: pd.Series,
    frame_dict: Dict[int, pd.DataFrame],
    available_frames: List[int],
    player_team_map: Dict[int, str],
    pitch_dims: Tuple[float, float] = (105.0, 68.0),
) -> Dict:
    """
    Robust, NaN-safe snapshot feature computation for a single event.

    - Handles missing player IDs and frames.
    - Detects centered coordinates and computes goal positions accordingly.
    - Deduplicates tracking rows per player, excludes non-detections, excludes self when counting teammates.
    - Includes max_player_targeted_xthreat and fills third_start/third_end when missing.
    - Computes dist_to_attacking_goal using event_row['attacking_side'] if present.
    Returns dict of features or {'error': <msg>}.
    """
    import pandas as pd
    import numpy as np
    import math
    from scipy.spatial import cKDTree

    # 1) choose anchor frame (safe)
    frame_candidate = None
    for col in ("frame_end", "frame_start"):
        val = event_row.get(col)
        if pd.notna(val):
            try:
                frame_candidate = int(val)
                break
            except Exception:
                continue
    if frame_candidate is None:
        val = event_row.get("possession_start")
        if pd.notna(val):
            try:
                frame_candidate = int(val)
            except Exception:
                frame_candidate = None
    if frame_candidate is None:
        return {"error": "no_frame_anchor"}

    # 2) nearest available frame (safe)
    try:
        chosen_frame = nearest_available_frame(frame_candidate, available_frames)
    except Exception:
        return {"error": "no_available_frames_for_match"}

    frame_df = frame_dict.get(chosen_frame)
    if frame_df is None or frame_df.shape[0] == 0:
        return {"error": "frame_missing"}

    # 3) NaN-safe player id selection
    pid = event_row.get("player_in_possession_id")
    ptid = event_row.get("player_targeted_id")
    if pd.notna(pid):
        try:
            rec_player_id = int(pid)
        except Exception:
            return {"error": "invalid_player_in_possession_id"}
    elif pd.notna(ptid):
        try:
            rec_player_id = int(ptid)
        except Exception:
            return {"error": "invalid_player_targeted_id"}
    else:
        return {"error": "no_player_id"}

    # 4) team short — prefer event team_shortname, else map from player_team_map
    rec_team_short = event_row.get("team_shortname")
    if pd.isna(rec_team_short) or rec_team_short is None:
        rec_team_short = player_team_map.get(rec_player_id, "UNKNOWN")

    # 5) prepare frame data safely
    frame_df = frame_df.copy()
    if "player_id" not in frame_df.columns or "x" not in frame_df.columns or "y" not in frame_df.columns:
        return {"error": "frame_missing_required_columns"}

    # coerce player_id safely
    try:
        frame_df["player_id"] = frame_df["player_id"].astype(int)
    except Exception:
        frame_df["player_id"] = pd.to_numeric(frame_df["player_id"], errors="coerce").fillna(-1).astype(int)

    # map team_shortname for each tracked player (use provided map; leave existing if mapping missing)
    frame_df["team_shortname"] = frame_df["player_id"].map(player_team_map).fillna(frame_df.get("team_shortname", "UNKNOWN"))

    # ensure x/y numeric
    frame_df["x"] = pd.to_numeric(frame_df["x"], errors="coerce")
    frame_df["y"] = pd.to_numeric(frame_df["y"], errors="coerce")

    # 6) player presence check
    player_mask = frame_df["player_id"] == rec_player_id
    if player_mask.sum() == 0:
        return {"error": "player_not_in_frame", "chosen_frame": int(chosen_frame), "rec_player_id": rec_player_id}

    px = float(frame_df.loc[player_mask, "x"].iloc[0])
    py = float(frame_df.loc[player_mask, "y"].iloc[0])

    # 7) detect coordinate origin (centered vs 0..pitch_length)
    pitch_length, pitch_width = pitch_dims
    centered_coords = (frame_df["x"].min() < 0) and (frame_df["x"].max() > 0)

    if centered_coords:
        half_len = pitch_length / 2.0
        goal_left = (-half_len, 0.0)
        goal_right = (half_len, 0.0)
        px_0based = px + half_len
    else:
        goal_left = (0.0, pitch_width / 2.0)
        goal_right = (pitch_length, pitch_width / 2.0)
        px_0based = px

    # 8) build a clean snapshot (deduplicate per player_id, remove invalid ids and optionally ghost rows)
    clean_snap = frame_df.copy()
    clean_snap = clean_snap[clean_snap["player_id"] >= 0]
    if "is_detected" in clean_snap.columns:
        clean_snap = clean_snap.sort_values(["player_id", "is_detected"], ascending=[True, False])
    else:
        clean_snap = clean_snap.sort_values("player_id")
    clean_snap = clean_snap.drop_duplicates(subset=["player_id"], keep="first")
    clean_snap["team_shortname"] = clean_snap["player_id"].map(player_team_map).fillna(clean_snap.get("team_shortname", "UNKNOWN"))

    # 9) separate teammates / opponents arrays (from clean snapshot)
    my_snap = clean_snap[clean_snap["team_shortname"] == rec_team_short]
    opp_snap = clean_snap[clean_snap["team_shortname"] != rec_team_short]

    my_pts = my_snap[["x", "y"]].to_numpy(dtype=float) if not my_snap.empty else np.empty((0, 2))
    opp_pts = opp_snap[["x", "y"]].to_numpy(dtype=float) if not opp_snap.empty else np.empty((0, 2))

    # 10) opponent KDTree metrics (safe)
    if opp_pts.shape[0] > 0:
        opp_tree = cKDTree(opp_pts)
        try:
            d_nearest_opp, _ = opp_tree.query([px, py], k=1)
        except Exception:
            d_nearest_opp = np.nan
        try:
            n_opp_within5 = len(opp_tree.query_ball_point([px, py], r=5.0))
        except Exception:
            n_opp_within5 = 0
    else:
        d_nearest_opp = np.nan
        n_opp_within5 = 0

    # 11) teammate metrics (safe) - use deduplicated my_snap (exclude self)
    if my_snap.shape[0] > 1:
        teammates = my_snap[my_snap["player_id"] != rec_player_id][["x", "y"]].to_numpy(dtype=float)
        if teammates.shape[0] > 0:
            try:
                my_tree = cKDTree(teammates)
                d_nearest_team, _ = my_tree.query([px, py], k=1)
            except Exception:
                d_nearest_team = np.nan
            mean_team_dist = float(np.mean(np.hypot(teammates[:, 0] - px, teammates[:, 1] - py)))
        else:
            d_nearest_team = np.nan
            mean_team_dist = np.nan
    else:
        d_nearest_team = np.nan
        mean_team_dist = np.nan

    # 12) forward options (deduped unique teammates ahead)
    try:
        teammates_ahead = my_snap[(my_snap["player_id"] != rec_player_id) & (my_snap["x"] > px)]
        n_forward_options = int(teammates_ahead["player_id"].nunique())
    except Exception:
        n_forward_options = 0
    if n_forward_options < 0:
        n_forward_options = 0
    if n_forward_options > 11:
        n_forward_options = min(n_forward_options, 11)

    # 13) distances to both goals (safe)
    dist_to_left_goal = math.hypot(px - goal_left[0], py - goal_left[1])
    dist_to_right_goal = math.hypot(px - goal_right[0], py - goal_right[1])
    dist_to_near_goal = float(min(dist_to_left_goal, dist_to_right_goal))
    dist_to_far_goal = float(max(dist_to_left_goal, dist_to_right_goal))

    # Determine attacking side from event_row (fall back to provider assumption if missing)
    attacking_side = event_row.get("attacking_side")
    if pd.isna(attacking_side) or attacking_side is None:
        # provider mirrors so team in possession attacks left->right by default in per-period mirroring;
        # if you want to be strict you can inspect event_row/team_in_possession, but default to left_to_right
        attacking_side = "left_to_right"

    # normalize the attacking_side string
    attacking_side = str(attacking_side).strip().lower()

    # NEW: distance to attacking goal based on attacking_side
    # attacking towards right -> attacking goal is right goal; attacking towards left -> left goal
    if attacking_side in ("left_to_right", "left_to_right " , "left_to_right".strip()):
        dist_to_attacking_goal = float(dist_to_right_goal)
    elif attacking_side in ("right_to_left", "right_to_left " , "right_to_left".strip()):
        dist_to_attacking_goal = float(dist_to_left_goal)
    else:
        # fallback: assume left_to_right
        dist_to_attacking_goal = float(dist_to_right_goal)

    # 14) read max_player_targeted_xthreat from event row (NaN-safe)
    xthreat_val = event_row.get("max_player_targeted_xthreat")
    try:
        max_player_targeted_xthreat = float(xthreat_val) if pd.notna(xthreat_val) else np.nan
    except Exception:
        max_player_targeted_xthreat = np.nan

    dangerous_val = event_row.get("player_targeted_dangerous")
    try:
        dangerous_val = float(dangerous_val) if pd.notna(dangerous_val) else np.nan
    except Exception:
        dangerous_val = np.nan

    dangerous_passes = event_row.get("n_passing_options_dangerous_not_difficult")
    try:
        dangerous_passes = float(dangerous_passes) if pd.notna(dangerous_passes) else np.nan
    except Exception:
        dangerous_passes = np.nan

    # 15) compute thirds if missing in event row (use px_0based and pitch thirds)
    def classify_third_from_x(x0):
        if x0 <= (pitch_length / 3.0):
            return "defensive"
        elif x0 <= (2.0 * pitch_length / 3.0):
            return "middle"
        else:
            return "attacking"

    third_start = event_row.get("third_start")
    third_end = event_row.get("third_end")
    if pd.isna(third_start) or third_start is None:
        try:
            third_start = classify_third_from_x(px_0based)
        except Exception:
            third_start = None
    if pd.isna(third_end) or third_end is None:
        x_end_val = event_row.get("x_end")
        if pd.notna(x_end_val):
            try:
                x_end = float(x_end_val)
                if centered_coords:
                    x_end_0 = x_end + (pitch_length / 2.0)
                else:
                    x_end_0 = x_end
                third_end = classify_third_from_x(x_end_0)
            except Exception:
                third_end = third_start
        else:
            third_end = third_start


    # 16) assemble feature dict with NaN-safe conversions
    features = {
        "Unique ID": event_row.get("Unique ID"),
        "match_id": parse_match_id_from_unique(event_row.get("Unique ID")),
        "event_index": int(event_row.get("index")) if pd.notna(event_row.get("index")) else None,
        "frame_anchor": int(chosen_frame),
        "rec_player_id": int(rec_player_id),
        "rec_team_short": rec_team_short,
        "dist_to_near_goal": dist_to_near_goal,
        "dist_to_far_goal": dist_to_far_goal,
        "dist_to_attacking_goal": dist_to_attacking_goal,        # <-- attacking-side aware
        "d_nearest_opp": float(d_nearest_opp) if not (d_nearest_opp is None) and not (isinstance(d_nearest_opp, float) and np.isnan(d_nearest_opp)) else np.nan,
        "n_opp_within5": int(n_opp_within5),
        "d_nearest_team": float(d_nearest_team) if not (d_nearest_team is None) and not (isinstance(d_nearest_team, float) and np.isnan(d_nearest_team)) else np.nan,
        "mean_team_dist": float(mean_team_dist) if not (mean_team_dist is None) and not (isinstance(mean_team_dist, float) and np.isnan(mean_team_dist)) else np.nan,
        "n_forward_options": int(n_forward_options),
        # context columns (safe get)
        "game_state": event_row.get("game_state"),
        "team_out_of_possession_phase_type": event_row.get("team_out_of_possession_phase_type"),
        "third_start": third_start,
        "third_end": third_end,
        "start_type": event_row.get("start_type"),
        "end_type": event_row.get("end_type"),
        # label included
        "max_player_targeted_xthreat": max_player_targeted_xthreat,
        "player_targeted_dangerous": dangerous_val,
        "n_passing_options_dangerous_not_difficult": dangerous_passes
    }

    return features


# Orchestration functions -----------------------------------------------------
def build_player_team_map_from_events(events_df: pd.DataFrame, match_id: int) -> Dict[int, str]:
    """
    Build a mapping from player_id -> team_shortname for the given match_id.
    We parse 'Unique ID' to filter events for the match and then map player ids seen in events.
    """
    # filter events for this match based on Unique ID parsing
    ev = events_df.copy()
    ev["match_id_parsed"] = ev["Unique ID"].apply(parse_match_id_from_unique)
    evm = ev[ev["match_id_parsed"] == match_id]
    # prefer player_in_possession_id -> team_shortname mapping
    mapping = {}
    # candidate columns that hold player ids and team names
    player_cols = [col for col in ["player_in_possession_id", "player_targeted_id", "player_id"] if col in evm.columns]
    for _, row in evm.iterrows():
        for pc in player_cols:
            pid = row.get(pc)
            tname = row.get("team_shortname") or row.get("team_in_possession_shortname")
            if not pd.isna(pid) and not pd.isna(tname):
                try:
                    mapping[int(pid)] = tname
                except Exception:
                    continue
    return mapping


def compute_features_for_all_events(events_csv: Path = EVENTS_CSV, tracking_csv: Path = TRACKING_CSV) -> pd.DataFrame:
    """
    Main entry point: read the two CSVs, iterate events, compute snapshot features, and return DataFrame.

    Steps:
      - loads event and tracking CSVs
      - builds helper structures (frames dict per match)
      - for each event extracts match_id, chooses frame list and frame dict, computes features
      - safely handles missing player IDs, frames, and errors
    """
    import pandas as pd

    print("Loading events...")
    events_df = pd.read_csv(events_csv)
    print("Loading tracking snapshot...")
    tracking_df = pd.read_csv(tracking_csv)

    # standardize types
    if "match_id" in tracking_df.columns:
        tracking_df["match_id"] = tracking_df["match_id"].astype(int)

    # Precompute mapping of available frames per match
    frames_map = group_tracking_by_match_and_frame(tracking_df)

    all_features = []

    for idx, ev in events_df.iterrows():
        unique = ev.get("Unique ID")
        match_id = parse_match_id_from_unique(unique)
        if match_id is None:
            continue  # skip if match_id cannot be parsed

        available_frames = frames_map.get(match_id, [])
        if not available_frames:
            continue  # skip matches without frames

        frame_dict = build_frames_dict(tracking_df, match_id)

        # --- NaN-safe player ID handling ---
        pid = ev.get("player_in_possession_id")
        ptid = ev.get("player_targeted_id")

        if pd.notna(pid):
            rec_player_id = int(pid)
        elif pd.notna(ptid):
            rec_player_id = int(ptid)
        else:
            # no valid player id, skip row
            feat = {"error": "no_player_id", "Unique ID": unique, "event_row_index": int(idx)}
            all_features.append(feat)
            continue

        # Build player->team map
        player_team_map = build_player_team_map_from_events(events_df, match_id)

        # Compute snapshot features
        feat = compute_snapshot_features_for_event(ev, frame_dict, available_frames, player_team_map)

        # Merge identifiers safely
        if isinstance(feat, dict):
            feat["Unique ID"] = unique
            feat["event_row_index"] = int(idx)
            feat["source_file"] = ev.get("source_file") if "source_file" in ev else None
            dangerous_cols = [
                "player_targeted_dangerous",
                "n_passing_options_dangerous_not_difficult",
                "max_player_targeted_xthreat",
            ]

            for col in dangerous_cols:
                raw = ev.get(col) if col in ev else np.nan
                if col not in feat or pd.isna(feat[col]):
                    try:
                        feat[col] = float(raw) if pd.notna(raw) else np.nan
                    except:
                        feat[col] = raw

        all_features.append(feat)
    for i, rec in enumerate(all_features):
        if isinstance(rec, dict):
            # ensure column exists for every dict record
            rec.setdefault("max_player_targeted_xthreat", np.nan)
            rec.setdefault("player_targeted_dangerous", np.nan)
            rec.setdefault("n_passing_options_dangerous_not_difficult", np.nan)
        else:
            # convert non-dict rows to minimal dict so DataFrame is consistent
            all_features[i] = {"error": "non_dict_record", "max_player_targeted_xthreat": np.nan}
    feats_df = pd.DataFrame(all_features)

    # optional: warn about rows with errors
    if "error" in feats_df.columns:
        errs = feats_df[feats_df["error"].notna()]
        if not errs.empty:
            print(f"Found {len(errs)} rows with errors (player/frame issues). They will remain flagged.")

    return feats_df



'''
def main():
    print("Running snapshot feature generation...")
    feats = compute_features_for_all_events()
    print("Saving to", OUT_FILE)
    feats.to_parquet(OUT_FILE, index=False)
    print("Done. Output rows:", len(feats))


if __name__ == "__main__":
    main()
'''