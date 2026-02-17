from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import streamlit as st
from streamlit_gsheets import GSheetsConnection

st.set_page_config(page_title="MLS Prediction Leaderboard", layout="wide")


def calculate_points(
    predicted_home: int,
    predicted_away: int,
    actual_home: Optional[int],
    actual_away: Optional[int],
) -> int:
    """Return scoring points for a prediction.

    Rules:
    - 3 points for exact scoreline.
    - 1 point for correct match result (home win / draw / away win).
    - 0 points otherwise.
    """
    if actual_home is None or actual_away is None:
        return 0

    if predicted_home == actual_home and predicted_away == actual_away:
        return 3

    predicted_result = (predicted_home > predicted_away) - (predicted_home < predicted_away)
    actual_result = (actual_home > actual_away) - (actual_home < actual_away)
    return 1 if predicted_result == actual_result else 0


def parse_kickoff(value: object) -> Optional[datetime]:
    """Convert a sheet value into a timezone-aware UTC datetime when possible."""
    if pd.isna(value):
        return None

    dt = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(dt):
        return None

    return dt.to_pydatetime()


def load_matches(conn: GSheetsConnection) -> pd.DataFrame:
    """Load upcoming fixtures from Google Sheets.

    Expected sheet columns:
      - match_id
      - home_team
      - away_team
      - match_kickoff
    """
    try:
        matches = conn.read(worksheet="fixtures", ttl=0)
        if matches is None or matches.empty:
            raise ValueError("No fixtures found")
    except Exception:
        # Sensible starter data so the app still runs before sheet wiring is complete.
        matches = pd.DataFrame(
            [
                {
                    "match_id": "MLS-001",
                    "home_team": "LA Galaxy",
                    "away_team": "Inter Miami",
                    "match_kickoff": "2099-07-01T23:30:00Z",
                },
                {
                    "match_id": "MLS-002",
                    "home_team": "Seattle Sounders",
                    "away_team": "Portland Timbers",
                    "match_kickoff": "2099-07-02T02:30:00Z",
                },
            ]
        )

    matches["match_kickoff"] = matches["match_kickoff"].apply(parse_kickoff)
    return matches


def submit_predictions(conn: GSheetsConnection, rows: list[dict]) -> None:
    """Append predictions to the predictions worksheet."""
    if not rows:
        return

    payload = pd.DataFrame(rows)
    try:
        existing = conn.read(worksheet="predictions", ttl=0)
        if existing is None:
            existing = pd.DataFrame()
    except Exception:
        existing = pd.DataFrame()

    combined = pd.concat([existing, payload], ignore_index=True)
    conn.update(worksheet="predictions", data=combined)


st.title("⚽ MLS Prediction Leaderboard")
st.caption("Submit your predictions before kickoff and track points in Google Sheets.")

with st.sidebar:
    st.header("User Login")
    user_name = st.text_input("Name", placeholder="Your display name")
    passcode = st.text_input("Passcode", type="password")
    is_logged_in = bool(user_name.strip()) and bool(passcode.strip())
    if is_logged_in:
        st.success("Logged in")
    else:
        st.info("Enter name and passcode to start")

conn = st.connection("gsheets", type=GSheetsConnection)
fixtures_df = load_matches(conn)
now_utc = datetime.now(timezone.utc)

if not is_logged_in:
    st.warning("Please log in from the sidebar to submit predictions.")
else:
    upcoming = fixtures_df[fixtures_df["match_kickoff"].notna()].copy()
    upcoming = upcoming.sort_values("match_kickoff")

    if upcoming.empty:
        st.info("No upcoming matches are currently available.")
    else:
        with st.form("prediction_form", clear_on_submit=False):
            st.subheader("Upcoming MLS Matches")
            prediction_rows: list[dict] = []

            for _, match in upcoming.iterrows():
                kickoff: datetime = match["match_kickoff"]
                is_open = now_utc < kickoff

                with st.container(border=True):
                    st.markdown(
                        f"**{match['home_team']} vs {match['away_team']}**  \\\nKickoff: {kickoff.strftime('%Y-%m-%d %H:%M UTC')}"
                    )

                    if is_open:
                        col1, col2 = st.columns(2)
                        pred_home = col1.number_input(
                            f"{match['home_team']} goals",
                            min_value=0,
                            max_value=15,
                            step=1,
                            key=f"home_{match['match_id']}",
                        )
                        pred_away = col2.number_input(
                            f"{match['away_team']} goals",
                            min_value=0,
                            max_value=15,
                            step=1,
                            key=f"away_{match['match_id']}",
                        )
                        prediction_rows.append(
                            {
                                "submitted_at": now_utc.isoformat(),
                                "user_name": user_name.strip(),
                                "match_id": match["match_id"],
                                "home_team": match["home_team"],
                                "away_team": match["away_team"],
                                "match_kickoff": kickoff.isoformat(),
                                "pred_home": int(pred_home),
                                "pred_away": int(pred_away),
                            }
                        )
                    else:
                        st.error("Predictions are closed for this match (kickoff has passed).")

            submitted = st.form_submit_button("Save Predictions")

        if submitted:
            if not prediction_rows:
                st.warning("No open matches available for prediction.")
            else:
                submit_predictions(conn, prediction_rows)
                st.success("Predictions saved to Google Sheets.")

st.divider()
st.markdown("### Points Logic")
st.code(
    """calculate_points(pred_home, pred_away, actual_home, actual_away)\n"
    "- 3 points for exact score\n"
    "- 1 point for correct result (W/D/L)\n"
    "- 0 otherwise""",
    language="text",
)
