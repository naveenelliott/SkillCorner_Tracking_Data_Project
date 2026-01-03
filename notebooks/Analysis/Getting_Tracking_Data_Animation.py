import pandas as pd
from mplsoccer import Pitch ,VerticalPitch
import json
import numpy as np
import warnings
from matplotlib import animation
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

results_df = pd.read_csv('Our Datasets/classifier_results_logreg.csv')

results_df.drop(columns={'Unique ID', 'event_index', 'source_file', 'row_index'}, inplace=True)

top_10_highest = results_df.sort_values('predicted_probability', ascending=False).head(10)

top_10_lowest = results_df.sort_values('predicted_probability', ascending=True).head(10)

tracking_data = pd.read_csv('Our Datasets/processed_tracking_data_for_test_match_ids.csv')

tracking_data.drop(columns={'is_detected_ball', 'frame_start', 'frame_end'}, inplace=True)

top_10_highest['rec_player_id'] = top_10_highest['rec_player_id'].astype(int)
top_10_highest['match_id'] = top_10_highest['match_id'].astype(int)

tracking_data['match_id'] = tracking_data['match_id'].astype(int)

tracking_data_highest = pd.merge(top_10_highest, tracking_data, left_on=['match_id', 'frame_start'], 
                                 right_on=['match_id', 'frame'], how
                                 ='outer')

tracking_data_lowest = pd.merge(top_10_lowest, tracking_data, left_on=['match_id', 'frame_start'], 
                                 right_on=['match_id', 'frame'], how
                                 ='outer')

tracking_data_highest_iterable = pd.merge(top_10_highest, tracking_data, left_on=['match_id', 'frame_start'], 
                                 right_on=['match_id', 'frame'], how
                                 ='inner')

tracking_data_highest_iterable = tracking_data_highest_iterable[['match_id', 'frame_start', 'frame_end', 'frame_start_rough_bound',
                                                                 'frame_end_rough_bound', 'team_out_of_possession_phase_type', 'predicted_probability']]

tracking_data_highest_iterable.drop_duplicates(inplace=True)


tracking_data_highest = tracking_data_highest.loc[(tracking_data_highest['frame'] >= 32474) & (tracking_data_highest['frame'] <= 32714)]

tracking_data_highest.reset_index(drop=True, inplace=True)


match_id = tracking_data_highest.loc[0, 'match_id']

def time_to_seconds(time_str):
    if time_str is None:
        return 90 * 60  # 120 minutes = 7200 seconds
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s

file_path = f"data/matches/{match_id}/{match_id}_match.json"

with open(file_path, "r") as f:
    raw_match_data = json.load(f)

# The output has nested json elements. We process them
raw_match_df = pd.json_normalize(raw_match_data, max_level=2)
raw_match_df["home_team_side"] = raw_match_df["home_team_side"].astype(str)

players_df = pd.json_normalize(
    raw_match_df.to_dict("records"),
    record_path="players",
    meta=[
        "home_team_score",
        "away_team_score",
        "date_time",
        "home_team_side",
        "home_team.name",
        "home_team.id",
        "away_team.name",
        "away_team.id",
    ],  # data we keep
)


# Take only players who played and create their total time
players_df = players_df[
    ~((players_df.start_time.isna()) & (players_df.end_time.isna()))
]

# Create a flag for GK
players_df["is_gk"] = players_df["player_role.acronym"] == "GK"

# Add a flag if the given player is home or away
players_df["match_name"] = (
    players_df["home_team.name"] + " vs " + players_df["away_team.name"]
)


# Add a flag if the given player is home or away
players_df["home_away_player"] = np.where(
    players_df.team_id == players_df["home_team.id"], "Home", "Away"
)

# Create flag from player
players_df["team_name"] = np.where(
    players_df.team_id == players_df["home_team.id"],
    players_df["home_team.name"],
    players_df["away_team.name"],
)

# Figure out sides
players_df[["home_team_side_1st_half", "home_team_side_2nd_half"]] = (
    players_df["home_team_side"]
    .astype(str)
    .str.strip("[]")
    .str.replace("'", "")
    .str.split(", ", expand=True)
)
# Clean up sides
players_df["direction_player_1st_half"] = np.where(
    players_df.home_away_player == "Home",
    players_df.home_team_side_1st_half,
    players_df.home_team_side_2nd_half,
)
players_df["direction_player_2nd_half"] = np.where(
    players_df.home_away_player == "Home",
    players_df.home_team_side_2nd_half,
    players_df.home_team_side_1st_half,
)


# Clean up and keep the columns that we want to keep about

columns_to_keep = [
    "match_name",
    "home_team.name",
    "away_team.name",
    "id",
    "short_name",
    "team_id",
    "team_name",
    "player_role.position_group",
    "player_role.name",
    "player_role.acronym",
    "is_gk",
    "direction_player_1st_half",
    "direction_player_2nd_half",
]
players_df = players_df[columns_to_keep]

enriched_tracking_data = tracking_data_highest.merge(
    players_df, left_on=["player_id"], right_on=["id"]
)

#enriched_tracking_data['rec_team_short'] = enriched_tracking_data['rec_team_short'] + ' ' + 'Football Club'

idx = enriched_tracking_data['rec_player_id'].first_valid_index()

recover_player_id = enriched_tracking_data.loc[idx, 'rec_player_id']


# team of the recovering player
recover_team = (
    enriched_tracking_data
        .loc[enriched_tracking_data['player_id'] == recover_player_id, 'team_name']
        .iloc[0]
)

# ball carrier flag (only that player)
enriched_tracking_data['ball_carrier'] = (
    enriched_tracking_data['player_id'] == recover_player_id
)

# teammates-in-possession flag
enriched_tracking_data['tip'] = (
    enriched_tracking_data['team_name'] == recover_team
)

df_away = enriched_tracking_data[enriched_tracking_data['tip'] == False]

df_home = enriched_tracking_data[enriched_tracking_data['tip'] == True]

df_ball = enriched_tracking_data[enriched_tracking_data.ball_carrier == True]

# First set up the figure, the axis
pitch = Pitch(pitch_type='skillcorner', goal_type='line', pitch_width=68, pitch_length=105)
fig, ax = pitch.draw(figsize=(16, 10.4))

# then setup the pitch plot markers we want to animate
marker_kwargs = {'marker': 'o', 'markeredgecolor': 'black', 'linestyle': 'None'}
ball, = ax.plot([], [], ms=6, markerfacecolor='w', zorder=3, **marker_kwargs)
away, = ax.plot([], [], ms=10, markerfacecolor='red', **marker_kwargs)  # red/maroon
home, = ax.plot([], [], ms=10, markerfacecolor='green', **marker_kwargs)  # purple
recover, = ax.plot([], [], ms=14, markerfacecolor='#00ff00',
                   zorder=5, **marker_kwargs)
arrow = ax.arrow(0, 0, 0, 0, color='black', width=0.25, zorder=4, length_includes_head=True)

ARROW_LEN = 0.5  # in pitch units (meters for SkillCorner)


df_ball = df_ball.sort_values("frame").reset_index(drop=True)

df_ball.drop_duplicates(subset=['frame'], inplace=True)

df_home['frame'] = df_home['frame'].astype(int)
df_away['frame'] = df_away['frame'].astype(int)
df_ball['frame'] = df_ball['frame'].astype(int)

def animate(i):
    frame = df_ball.iloc[i]['frame']

    # --- ball position ---
    bx = df_ball.iloc[i]['ball_x']
    by = df_ball.iloc[i]['ball_y']
    ball.set_data([bx], [by])

    # players
    away_frame = df_away[df_away.frame == frame]
    home_frame = df_home[df_home.frame == frame]

    away.set_data(list(away_frame['x']), list(away_frame['y']))
    home.set_data(list(home_frame['x']), list(home_frame['y']))
    
    recover.set_data(
        [df_ball.iloc[i]['x']],
        [df_ball.iloc[i]['y']]
    )
    
    # --- DIRECTION ARROW ---
    global arrow
    if i > 0:
        px = df_ball.iloc[i-1]['ball_x']
        py = df_ball.iloc[i-1]['ball_y']
 
        dx = bx - px
        dy = by - py
 
        # remove old arrow
        arrow.remove()
        
        # -------- normalize length --------
        mag = np.hypot(dx, dy)
        if mag > 0:
            dx = dx / mag * ARROW_LEN
            dy = dy / mag * ARROW_LEN
 
        # draw new arrow
        arrow = ax.arrow(px, py, dx, dy,
                         color='black',
                         width=0.25,
                         length_includes_head=True,
                         zorder=4)

    return ball, away, home, recover, arrow


# 👇 KEEP ANIMATION OBJECT IN A VARIABLE THAT PERSISTS
anim = FuncAnimation(
    fig,
    animate,
    frames=len(df_ball),
    interval=100,        # 1 second per frame (for display)
    blit=False,
    repeat=False
)

# 👇 SAVE AFTER DISPLAY IS CREATED
writer = PillowWriter(fps=5)      # 1 frame per second in the GIF too
anim.save("notebooks/Analysis/goal_animation.gif", writer=writer, dpi=120)
