from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests
import streamlit as st
from streamlit_gsheets import GSheetsConnection

st.set_page_config(page_title="Atlanta United Prediction League", layout="wide")

USERS_FILE = Path(__file__).with_name("users.json")
SCHEDULE_FILE = Path(__file__).with_name("schedule.json")
API_BASE_URL = "https://www.thesportsdb.com/api/v1/json/123"
MLS_LEAGUE_ID = 4346
MLS_SEASON = "2026"
ATLANTA_TEAM_NAME = "Atlanta United"


def load_users() -> dict[str, str]:
    """Load users and passcodes from the local JSON file."""
    if not USERS_FILE.exists():
        return {}

    with USERS_FILE.open("r", encoding="utf-8") as infile:
        users = json.load(infile)

    if isinstance(users, dict):
        return {str(name): str(passcode) for name, passcode in users.items()}

    return {}


def save_users(users: dict[str, str]) -> None:
    """Persist users and passcodes to disk."""
    with USERS_FILE.open("w", encoding="utf-8") as outfile:
        json.dump(users, outfile, indent=2)


def calculate_points(
    predicted_home: int,
    predicted_away: int,
    actual_home: Optional[int],
    actual_away: Optional[int],
) -> int:
    """Return scoring points for a prediction."""
    if actual_home is None or actual_away is None:
        return 0

    if predicted_home == actual_home and predicted_away == actual_away:
        return 3

    predicted_result = (predicted_home > predicted_away) - (predicted_home < predicted_away)
    actual_result = (actual_home > actual_away) - (actual_home < actual_away)
    return 1 if predicted_result == actual_result else 0


def parse_kickoff(value: object) -> Optional[datetime]:
    """Convert a sheet/API value into a timezone-aware UTC datetime when possible."""
    if pd.isna(value):
        return None

    dt = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(dt):
        return None

    return dt.to_pydatetime()


def _as_int(value: object) -> Optional[int]:
    num = pd.to_numeric(value, errors="coerce")
    if pd.isna(num):
        return None
    return int(num)


def load_mls_atlanta_fixtures() -> pd.DataFrame:
    """Load the full Atlanta United MLS season schedule from API, with local fallback."""
    # Use TheSportsDB "events by season" endpoint to load the complete MLS schedule.
    url = f"{API_BASE_URL}/eventsseason.php?id={MLS_LEAGUE_ID}&s={MLS_SEASON}"

    def fixture_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for fixture in events:
            home_name = fixture.get("strHomeTeam")
            away_name = fixture.get("strAwayTeam")
            if ATLANTA_TEAM_NAME not in (home_name, away_name):
                continue

            kickoff_date = fixture.get("dateEvent")
            kickoff_time = fixture.get("strTime")
            kickoff_value = f"{kickoff_date} {kickoff_time}" if kickoff_time else kickoff_date

            rows.append(
                {
                    "match_id": str(fixture.get("idEvent") or ""),
                    "home_team": home_name,
                    "away_team": away_name,
                    "match_kickoff": parse_kickoff(kickoff_value),
                    "final_home": _as_int(fixture.get("intHomeScore")),
                    "final_away": _as_int(fixture.get("intAwayScore")),
                }
            )

        return rows

    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        st.error(f"Failed to fetch fixtures from TheSportsDB: {exc}")
        payload = {"events": []}

    rows = fixture_rows(payload.get("events") or [])

    if not rows and SCHEDULE_FILE.exists():
        try:
            with SCHEDULE_FILE.open("r", encoding="utf-8") as infile:
                local_events = json.load(infile)
            if isinstance(local_events, list):
                rows = fixture_rows(local_events)
        except Exception as exc:
            st.warning(f"Unable to read local fallback schedule.json: {exc}")

    fixtures_df = pd.DataFrame(rows)
    if fixtures_df.empty:
        return fixtures_df

    return fixtures_df.sort_values("match_kickoff").reset_index(drop=True)


def load_predictions(conn: GSheetsConnection) -> pd.DataFrame:
    try:
        predictions = conn.read(worksheet="predictions", ttl=0)
        if predictions is None:
            return pd.DataFrame()
        return predictions
    except Exception:
        return pd.DataFrame()


def save_predictions(conn: GSheetsConnection, predictions: pd.DataFrame) -> None:
    conn.update(worksheet="predictions", data=predictions)


def upsert_user_prediction(conn: GSheetsConnection, row: dict[str, Any]) -> None:
    predictions = load_predictions(conn)

    if predictions.empty:
        updated = pd.DataFrame([row])
    else:
        required_cols = [
            "submitted_at",
            "user_name",
            "match_id",
            "home_team",
            "away_team",
            "match_kickoff",
            "pred_home",
            "pred_away",
            "points_earned",
            "final_home",
            "final_away",
        ]
        for col in required_cols:
            if col not in predictions.columns:
                predictions[col] = pd.NA

        mask = (
            predictions["user_name"].astype(str).eq(str(row["user_name"]))
            & predictions["match_id"].astype(str).eq(str(row["match_id"]))
        )
        predictions = predictions.loc[~mask].copy()
        updated = pd.concat([predictions, pd.DataFrame([row])], ignore_index=True)

    save_predictions(conn, updated)


def determine_prediction_target(fixtures: pd.DataFrame, now_utc: datetime) -> tuple[Optional[pd.Series], Optional[str]]:
    """Return one next-match target and optional lock message."""
    if fixtures.empty:
        return None, "No Atlanta United MLS fixtures were returned from the API."

    with_kickoff = fixtures[fixtures["match_kickoff"].notna()].copy()
    with_kickoff = with_kickoff.sort_values("match_kickoff")
    if with_kickoff.empty:
        return None, "No fixture kickoff timestamps are available."

    today = now_utc.date()
    if "status_short" in with_kickoff.columns:
        today_matches = with_kickoff[with_kickoff["match_kickoff"].dt.date == today]
        unresolved_today = today_matches[~today_matches["status_short"].astype(str).eq("FT")]

        if not unresolved_today.empty:
            current_match = unresolved_today.sort_values("match_kickoff").iloc[0]
            status_long = current_match.get("status_long") or "In progress"
            return (
                None,
                f"Predictions are locked until today's match is final (FT): "
                f"{current_match['home_team']} vs {current_match['away_team']} ({status_long}).",
            )

    upcoming = with_kickoff[with_kickoff["match_kickoff"] > now_utc]
    if upcoming.empty:
        return None, "No future Atlanta United matches are currently available."

    return upcoming.iloc[0], None


def auto_score_latest_finished_match(conn: GSheetsConnection, fixtures_df: pd.DataFrame, now_utc: datetime) -> Optional[pd.DataFrame]:
    """Find finished Atlanta United matches, score latest one, and return recent results."""
    del fixtures_df

    url = f"{API_BASE_URL}/eventspastleague.php"
    params = {"id": MLS_LEAGUE_ID}

    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        st.error(f"Failed to fetch completed matches from TheSportsDB: {exc}")
        return None

    rows: list[dict[str, Any]] = []
    for event in payload.get("events") or []:
        home_name = event.get("strHomeTeam")
        away_name = event.get("strAwayTeam")
        if ATLANTA_TEAM_NAME not in (home_name, away_name):
            continue

        home_score = _as_int(event.get("intHomeScore"))
        away_score = _as_int(event.get("intAwayScore"))
        if home_score is None or away_score is None:
            continue

        match_date = event.get("dateEvent")
        match_time = event.get("strTime")
        kickoff_value = f"{match_date} {match_time}" if match_time else match_date
        kickoff = parse_kickoff(kickoff_value)
        if kickoff is None or kickoff > now_utc:
            continue

        rows.append(
            {
                "match_id": str(event.get("idEvent")),
                "home_team": home_name,
                "away_team": away_name,
                "match_kickoff": kickoff,
                "final_home": home_score,
                "final_away": away_score,
            }
        )

    if not rows:
        return None

    results_df = pd.DataFrame(rows).sort_values("match_kickoff").reset_index(drop=True)
    latest = results_df.iloc[-1]
    actual_home = _as_int(latest["final_home"])
    actual_away = _as_int(latest["final_away"])
    if actual_home is None or actual_away is None:
        return None

    predictions = load_predictions(conn)
    if predictions.empty:
        return results_df.sort_values("match_kickoff", ascending=False).head(10).reset_index(drop=True)

    for col in ["pred_home", "pred_away", "match_id", "points_earned", "final_home", "final_away"]:
        if col not in predictions.columns:
            predictions[col] = pd.NA

    predictions["pred_home"] = pd.to_numeric(predictions["pred_home"], errors="coerce")
    predictions["pred_away"] = pd.to_numeric(predictions["pred_away"], errors="coerce")

    target_match_id = str(latest["match_id"])
    target_rows = predictions["match_id"].astype(str).eq(target_match_id)

    if target_rows.any():
        def row_points(row: pd.Series) -> Optional[int]:
            if pd.isna(row.get("pred_home")) or pd.isna(row.get("pred_away")):
                return pd.NA
            return calculate_points(int(row["pred_home"]), int(row["pred_away"]), actual_home, actual_away)

        predictions.loc[target_rows, "points_earned"] = predictions.loc[target_rows].apply(row_points, axis=1)
        predictions.loc[target_rows, "final_home"] = actual_home
        predictions.loc[target_rows, "final_away"] = actual_away
        save_predictions(conn, predictions)

    return results_df.sort_values("match_kickoff", ascending=False).head(10).reset_index(drop=True)


def format_countdown(kickoff: datetime, now_utc: datetime) -> str:
    remaining = kickoff - now_utc
    if remaining.total_seconds() <= 0:
        return "Kickoff reached"
    days = remaining.days
    hours, rem = divmod(remaining.seconds, 3600)
    mins, _ = divmod(rem, 60)
    return f"{days}d {hours}h {mins}m"


st.title("⚽ Atlanta United MLS Prediction League")
st.caption("Live fixtures via TheSportsDB V1 API with auto-scoring in Google Sheets.")

with st.sidebar:
    st.header("User Access")
    st.session_state.setdefault("logged_in_user", None)

    login_tab, signup_tab = st.tabs(["Login", "Sign Up"])

    with login_tab:
        user_name = st.text_input("Name", placeholder="Your display name")
        passcode = st.text_input("Passcode", type="password")
        login_clicked = st.button("Log In", type="primary")

        if login_clicked:
            users = load_users()
            entered_name = user_name.strip()
            entered_passcode = passcode.strip()

            if users.get(entered_name) == entered_passcode and entered_name:
                st.session_state["logged_in_user"] = entered_name
                st.success("Logged in")
            else:
                st.error("Invalid name or passcode")

        if st.session_state["logged_in_user"]:
            st.caption(f"Logged in as **{st.session_state['logged_in_user']}**")
            if st.button("Log Out"):
                st.session_state["logged_in_user"] = None
                st.info("Logged out")

    with signup_tab:
        new_user_name = st.text_input("New username", key="signup_name")
        new_passcode = st.text_input("New passcode", type="password", key="signup_pass")
        create_account_clicked = st.button("Create Account")

        if create_account_clicked:
            entered_name = new_user_name.strip()
            entered_passcode = new_passcode.strip()

            if not entered_name or not entered_passcode:
                st.error("Please enter both a username and passcode.")
            else:
                users = load_users()
                if entered_name in users:
                    st.error("Username already taken. Please pick another one.")
                else:
                    users[entered_name] = entered_passcode
                    save_users(users)
                    st.success("Account created! You can now log in.")

    is_logged_in = bool(st.session_state["logged_in_user"])
    active_user_name = st.session_state["logged_in_user"] or ""

conn = st.connection("gsheets", type=GSheetsConnection)
fixtures_df = load_mls_atlanta_fixtures()
now_utc = datetime.now(timezone.utc)
recent_results_df = auto_score_latest_finished_match(conn, fixtures_df, now_utc)
next_match, lock_message = determine_prediction_target(fixtures_df, now_utc)
predictions_df = load_predictions(conn)

matches_tab, leaderboard_tab, predictions_tab = st.tabs(["📅 Matches", "🏆 Leaderboard", "🔮 Predictions"])

with matches_tab:
    st.markdown(f"### {MLS_SEASON} Atlanta United Full Schedule")
    if fixtures_df.empty:
        st.info("No Atlanta United fixtures are currently available.")
    else:
        matches_display = fixtures_df.sort_values("match_kickoff").copy()

        if next_match is not None:
            matches_display["is_next_match"] = matches_display["match_id"].astype(str).eq(str(next_match["match_id"]))
        else:
            matches_display["is_next_match"] = False

        def match_status(row: pd.Series) -> str:
            kickoff = row.get("match_kickoff")
            final_home = row.get("final_home")
            final_away = row.get("final_away")

            if pd.notna(final_home) and pd.notna(final_away):
                return f"{row['home_team']} {int(final_home)} - {int(final_away)} {row['away_team']}"

            if pd.notna(kickoff):
                kickoff_dt = pd.Timestamp(kickoff).to_pydatetime()
                if kickoff_dt <= now_utc:
                    return "Match completed (score pending)"
                kickoff_str = kickoff_dt.strftime("%Y-%m-%d %H:%M UTC")
                if bool(row.get("is_next_match")):
                    return f"Kickoff: {kickoff_str} • Predictions Open"
                return f"Kickoff: {kickoff_str}"

            return "Kickoff TBD"

        matches_display["Status"] = matches_display.apply(match_status, axis=1)
        matches_display["Date (UTC)"] = matches_display["match_kickoff"].apply(
            lambda dt: dt.strftime("%Y-%m-%d %H:%M UTC") if pd.notna(dt) else "TBD"
        )

        st.dataframe(
            matches_display[["Date (UTC)", "home_team", "away_team", "Status"]].rename(
                columns={"home_team": "Home Team", "away_team": "Away Team"}
            ).reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )

with leaderboard_tab:
    st.markdown("### Season Standings")
    if predictions_df.empty or "user_name" not in predictions_df.columns:
        st.info("No prediction data available for leaderboard standings yet.")
    else:
        leaderboard = predictions_df.copy()
        if "points_earned" not in leaderboard.columns:
            leaderboard["points_earned"] = 0

        leaderboard["points_earned"] = pd.to_numeric(leaderboard["points_earned"], errors="coerce").fillna(0)
        standings = (
            leaderboard.groupby("user_name", dropna=True, as_index=False)["points_earned"]
            .sum()
            .sort_values("points_earned", ascending=False)
            .reset_index(drop=True)
        )

        if standings.empty:
            st.info("No leaderboard entries available yet.")
        else:
            standings["Rank"] = standings.index + 1
            standings = standings.rename(columns={"user_name": "User", "points_earned": "Total Points"})
            st.dataframe(standings[["Rank", "User", "Total Points"]], use_container_width=True, hide_index=True)

with predictions_tab:
    if next_match is not None:
        kickoff = next_match["match_kickoff"]
        st.markdown("## ⏳ Countdown to Kickoff")
        st.markdown(
            f"<h1 style='text-align:center;'>{format_countdown(kickoff, now_utc)}</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='text-align:center;font-size:1.2rem;'><strong>{next_match['home_team']} vs {next_match['away_team']}</strong><br>{kickoff.strftime('%Y-%m-%d %H:%M UTC')}</div>",
            unsafe_allow_html=True,
        )

    st.divider()
    if not is_logged_in:
        st.warning("Please log in from the sidebar to submit a prediction.")
    elif lock_message:
        st.warning(lock_message)
    elif next_match is None:
        st.info("No match currently open for prediction.")
    else:
        with st.form("prediction_form", clear_on_submit=False):
            st.subheader("Prediction Window: Next Atlanta United Match Only")
            st.write(
                f"**{next_match['home_team']} vs {next_match['away_team']}**\n"
                f"Kickoff: {next_match['match_kickoff'].strftime('%Y-%m-%d %H:%M UTC')}"
            )

            col1, col2 = st.columns(2)
            pred_home = col1.number_input(
                f"{next_match['home_team']} goals",
                min_value=0,
                max_value=15,
                step=1,
                key=f"home_{next_match['match_id']}",
            )
            pred_away = col2.number_input(
                f"{next_match['away_team']} goals",
                min_value=0,
                max_value=15,
                step=1,
                key=f"away_{next_match['match_id']}",
            )

            submitted = st.form_submit_button("Save Prediction")

        if submitted:
            if now_utc >= next_match["match_kickoff"]:
                st.error("Predictions are closed for this match (kickoff has passed).")
            else:
                row = {
                    "submitted_at": now_utc.isoformat(),
                    "user_name": active_user_name,
                    "match_id": str(next_match["match_id"]),
                    "home_team": next_match["home_team"],
                    "away_team": next_match["away_team"],
                    "match_kickoff": next_match["match_kickoff"].isoformat(),
                    "pred_home": int(pred_home),
                    "pred_away": int(pred_away),
                    "points_earned": pd.NA,
                    "final_home": pd.NA,
                    "final_away": pd.NA,
                }
                upsert_user_prediction(conn, row)
                st.success("Prediction saved to Google Sheets.")

    st.divider()
    st.markdown("### Points Logic")
    st.code(
        """calculate_points(pred_home, pred_away, actual_home, actual_away)
- 3 points for exact score
- 1 point for correct result (W/D/L)
- 0 otherwise""",
        language="text",
    )
