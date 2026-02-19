from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st
from streamlit_gsheets import GSheetsConnection
from streamlit_gsheets.gsheets_connection import GSheetsServiceAccountClient

st.set_page_config(page_title="Atlanta United Prediction League", layout="wide")

SCHEDULE_FILE = Path(__file__).with_name("schedule.json")
API_BASE_URL = "https://www.thesportsdb.com/api/v1/json/123"
MLS_LEAGUE_ID = 4346
MLS_SEASON = "2026"
ATLANTA_TEAM_NAME = "Atlanta United"
EASTERN_TZ = ZoneInfo("America/New_York")


def load_users() -> dict[str, str]:
    """Load users and passcodes from the users worksheet in Google Sheets."""
    service_account_client = _build_service_account_client()

    try:
        if service_account_client is not None:
            users_df = service_account_client.read(worksheet="users", ttl=0)
        else:
            conn = st.connection("gsheets", type=GSheetsConnection)
            users_df = conn.read(worksheet="users", ttl=0)
    except Exception:
        return {}

    if users_df is None or users_df.empty:
        return {}

    if "user_name" not in users_df.columns or "passcode" not in users_df.columns:
        return {}

    users: dict[str, str] = {}
    for _, row in users_df.iterrows():
        name = str(row.get("user_name") or "").strip()
        passcode = str(row.get("passcode") or "").strip()
        if name:
            users[name] = passcode

    return users


def save_users(users: dict[str, str]) -> None:
    """Persist users and passcodes to the users worksheet in Google Sheets."""
    users_df = pd.DataFrame(
        [{"user_name": str(name), "passcode": str(passcode)} for name, passcode in users.items()],
        columns=["user_name", "passcode"],
    )

    service_account_client = _build_service_account_client()
    if service_account_client is not None:
        service_account_client.update(worksheet="users", data=users_df)
        return

    conn = st.connection("gsheets", type=GSheetsConnection)
    conn.update(worksheet="users", data=users_df)


def calculate_points(
    predicted_home: int,
    predicted_away: int,
    actual_home: Optional[int],
    actual_away: Optional[int],
) -> int:
    """Return scoring points for a prediction."""
    if actual_home is None or actual_away is None:
        return 0

    predicted_home_match = predicted_home == actual_home
    predicted_away_match = predicted_away == actual_away

    if predicted_home == actual_home and predicted_away == actual_away:
        return 3

    predicted_result = (predicted_home > predicted_away) - (predicted_home < predicted_away)
    actual_result = (actual_home > actual_away) - (actual_home < actual_away)
    correct_result = predicted_result == actual_result
    has_partial_score_match = predicted_home_match or predicted_away_match

    if correct_result and has_partial_score_match:
        return 2
    if correct_result:
        return 1
    if has_partial_score_match:
        return 0
    return -1


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
    schedule_df = load_schedule_from_json()

    def match_signature(home_team: Any, away_team: Any, kickoff: Optional[datetime]) -> tuple[str, str, str]:
        kickoff_key = kickoff.isoformat() if kickoff is not None else ""
        return (str(home_team or ""), str(away_team or ""), kickoff_key)

    schedule_match_id_map: dict[str, str] = {}
    schedule_signature_map: dict[tuple[str, str, str], str] = {}
    if not schedule_df.empty and {"match_id", "home_team", "away_team", "match_kickoff"}.issubset(schedule_df.columns):
        for _, scheduled_row in schedule_df.iterrows():
            schedule_match_id = str(scheduled_row.get("match_id") or "")
            schedule_match_id_map[schedule_match_id] = schedule_match_id
            schedule_signature_map[
                match_signature(
                    scheduled_row.get("home_team"),
                    scheduled_row.get("away_team"),
                    parse_kickoff(scheduled_row.get("match_kickoff")),
                )
            ] = schedule_match_id

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
            kickoff_utc = parse_kickoff(kickoff_value)
            api_match_id = str(fixture.get("idEvent") or "")
            signature = match_signature(home_name, away_name, kickoff_utc)
            resolved_match_id = schedule_match_id_map.get(api_match_id) or schedule_signature_map.get(signature) or api_match_id

            rows.append(
                {
                    "match_id": str(resolved_match_id),
                    "home_team": home_name,
                    "away_team": away_name,
                    "match_kickoff": kickoff_utc,
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

    fixtures_df["match_id"] = fixtures_df["match_id"].astype(str)

    return fixtures_df.sort_values("match_kickoff").reset_index(drop=True)


def load_schedule_from_json() -> pd.DataFrame:
    """Load the full season schedule from schedule.json (kickoff times in ET)."""
    if not SCHEDULE_FILE.exists():
        return pd.DataFrame()

    with SCHEDULE_FILE.open("r", encoding="utf-8") as infile:
        events = json.load(infile)

    if not isinstance(events, list):
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for fixture in events:
        kickoff_date = fixture.get("dateEvent")
        kickoff_time = fixture.get("strTime")
        kickoff_value = f"{kickoff_date} {kickoff_time}" if kickoff_time else kickoff_date

        kickoff_local = pd.to_datetime(kickoff_value, errors="coerce")
        if pd.isna(kickoff_local):
            kickoff_utc = None
            kickoff_et_label = "TBD"
        else:
            kickoff_local = kickoff_local.tz_localize(EASTERN_TZ)
            kickoff_utc = kickoff_local.tz_convert(timezone.utc).to_pydatetime()
            kickoff_et_label = kickoff_local.strftime("%Y-%m-%d %I:%M %p ET")

        rows.append(
            {
                "match_id": str(fixture.get("idEvent") or ""),
                "segment": _as_int(fixture.get("segment")),
                "home_team": fixture.get("strHomeTeam"),
                "away_team": fixture.get("strAwayTeam"),
                "match_kickoff": kickoff_utc,
                "kickoff_et": kickoff_et_label,
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("match_kickoff").reset_index(drop=True)


def load_predictions(conn: GSheetsConnection) -> pd.DataFrame:
    service_account_client = _build_service_account_client()
    try:
        if service_account_client is not None:
            predictions = service_account_client.read(worksheet="predictions", ttl=0)
        else:
            predictions = conn.read(worksheet="predictions", ttl=0)
        if predictions is None:
            return pd.DataFrame()
        return predictions
    except Exception:
        return pd.DataFrame()


def save_predictions(conn: GSheetsConnection, predictions: pd.DataFrame) -> None:
    service_account_client = _build_service_account_client()
    if service_account_client is not None:
        service_account_client.update(worksheet="predictions", data=predictions)
        return
    conn.update(worksheet="predictions", data=predictions)


def _build_service_account_client() -> Optional[GSheetsServiceAccountClient]:
    try:
        has_service_account = "gcp_service_account" in st.secrets
    except Exception:
        return None

    if not has_service_account:
        return None

    service_account_secrets = dict(st.secrets["gcp_service_account"])
    if not service_account_secrets:
        return None

    merged_secrets: dict[str, Any] = {"type": "service_account", **service_account_secrets}

    connection_settings: dict[str, Any] = {}
    if "connections" in st.secrets and "gsheets" in st.secrets["connections"]:
        connection_settings = dict(st.secrets["connections"]["gsheets"])
    elif "gsheets" in st.secrets:
        connection_settings = dict(st.secrets["gsheets"])

    for key in ("spreadsheet", "url", "worksheet"):
        if key in connection_settings and key not in merged_secrets:
            merged_secrets[key] = connection_settings[key]

    try:
        return GSheetsServiceAccountClient(merged_secrets)
    except Exception:
        return None


def upsert_user_prediction(conn: GSheetsConnection, row: dict[str, Any]) -> None:
    predictions = load_predictions(conn)
    row_match_id = str(row["match_id"])

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

        mask = (predictions["user_name"].astype(str) == str(row["user_name"])) & (
            predictions["match_id"].astype(str) == row_match_id
        )
        predictions = predictions.loc[~mask].copy()
        updated = pd.concat([predictions, pd.DataFrame([row])], ignore_index=True)

    updated["user_name"] = updated["user_name"].astype(str)
    updated["match_id"] = updated["match_id"].astype(str)
    updated = updated.drop_duplicates(subset=["user_name", "match_id"], keep="last").reset_index(drop=True)

    save_predictions(conn, updated)


def determine_prediction_target(fixtures: pd.DataFrame, now_utc: datetime) -> tuple[Optional[pd.Series], Optional[str]]:
    """Return one next-match target and optional lock message."""
    if fixtures.empty:
        return None, "No Atlanta United MLS fixtures were returned from the API."

    with_kickoff = fixtures[fixtures["match_kickoff"].notna()].copy()
    with_kickoff = with_kickoff.sort_values("match_kickoff")
    if with_kickoff.empty:
        return None, "No fixture kickoff timestamps are available."

    now_et = now_utc.astimezone(EASTERN_TZ)
    today_et = now_et.date()
    today_matches = with_kickoff[with_kickoff["match_kickoff"].dt.tz_convert(EASTERN_TZ).dt.date == today_et]
    if not today_matches.empty:
        current_match = today_matches.sort_values("match_kickoff").iloc[0]
        prediction_deadline = current_match["match_kickoff"] + timedelta(minutes=15)
        if now_utc <= prediction_deadline:
            return current_match, None
        return (
            None,
            "Predictions for today's match are now closed. Come back tomorrow to predict the next game!",
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
schedule_df = load_schedule_from_json()
now_utc = datetime.now(timezone.utc)
recent_results_df = auto_score_latest_finished_match(conn, fixtures_df, now_utc)
next_match, lock_message = determine_prediction_target(fixtures_df, now_utc)
predictions_df = load_predictions(conn)
if not predictions_df.empty and "match_id" in predictions_df.columns:
    predictions_df["match_id"] = predictions_df["match_id"].astype(str)

match_segments = pd.DataFrame(columns=["match_id", "segment"])
if not schedule_df.empty and {"match_id", "segment"}.issubset(schedule_df.columns):
    match_segments = schedule_df[["match_id", "segment"]].copy().reset_index(drop=True)

if not match_segments.empty:
    match_segments["match_id"] = match_segments["match_id"].astype(str)
    match_segments["segment"] = pd.to_numeric(match_segments["segment"], errors="coerce").astype("Int64")

if predictions_df.empty or "user_name" not in predictions_df.columns:
    standings = pd.DataFrame(columns=["Rank", "User", "Total Points"])
    active_segment_standings = pd.DataFrame(columns=["Rank", "User", "Segment Points"])
    active_segment_label: Optional[str] = None
else:
    leaderboard = predictions_df.copy()
    for col in ["points_earned", "final_home", "final_away", "match_id"]:
        if col not in leaderboard.columns:
            leaderboard[col] = pd.NA

    leaderboard["points_earned"] = pd.to_numeric(leaderboard["points_earned"], errors="coerce").fillna(0)
    leaderboard["match_id"] = leaderboard["match_id"].astype(str)

    if not match_segments.empty:
        leaderboard = leaderboard.merge(match_segments, on="match_id", how="left")
    else:
        leaderboard["segment"] = pd.NA

    standings = (
        leaderboard.groupby("user_name", dropna=True, as_index=False)["points_earned"]
        .sum()
        .sort_values("points_earned", ascending=False)
        .reset_index(drop=True)
    )
    standings["Rank"] = standings.index + 1
    standings = standings.rename(columns={"user_name": "User", "points_earned": "Total Points"})

    completed_rows = leaderboard[leaderboard["final_home"].notna() & leaderboard["final_away"].notna()].copy()
    completed_match_ids = set(completed_rows["match_id"].dropna().astype(str).tolist())

    active_segment_label = None
    active_segment_standings = pd.DataFrame(columns=["Rank", "User", "Segment Points"])

    if not match_segments.empty:
        active_segment = pd.NA
        for segment_value in sorted(match_segments["segment"].dropna().unique()):
            segment_match_ids = set(match_segments[match_segments["segment"] == segment_value]["match_id"].tolist())
            if segment_match_ids - completed_match_ids:
                active_segment = segment_value
                break

        if pd.isna(active_segment):
            all_segment_values = sorted(match_segments["segment"].dropna().unique())
            if all_segment_values:
                active_segment = all_segment_values[-1]

        if pd.notna(active_segment):
            active_segment_label = f"Segment {int(active_segment)}"
            scored_segment_rows = completed_rows[completed_rows["segment"] == active_segment].copy()
            if not scored_segment_rows.empty:
                active_segment_standings = (
                    scored_segment_rows.groupby("user_name", dropna=True, as_index=False)["points_earned"]
                    .sum()
                    .sort_values("points_earned", ascending=False)
                    .reset_index(drop=True)
                )
                active_segment_standings["Rank"] = active_segment_standings.index + 1
                active_segment_standings = active_segment_standings.rename(
                    columns={"user_name": "User", "points_earned": "Segment Points"}
                )

home_tab, matches_tab, leaderboard_tab, predictions_tab, faq_tab = st.tabs(
    ["🏠 Home", "📅 Matches", "🏆 Leaderboard", "🔮 Predictions", "❓ FAQ"]
)

with home_tab:
    st.markdown("### Dashboard")

    st.markdown("#### ⏳ Countdown to Kickoff")
    if next_match is None:
        st.info("No future Atlanta United match is currently available from TheSportsDB.")
    else:
        kickoff = next_match["match_kickoff"]
        st.markdown(
            f"<h1 style='text-align:center;'>{format_countdown(kickoff, now_utc)}</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='text-align:center;font-size:1.2rem;'><strong>{next_match['home_team']} vs {next_match['away_team']}</strong><br>{kickoff.strftime('%Y-%m-%d %H:%M UTC')}</div>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("#### 🏆 Standings Snapshot (Top 3)")
    if standings.empty:
        st.info("No leaderboard data available yet.")
    else:
        st.dataframe(standings[["Rank", "User", "Total Points"]].head(3), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("#### 🕒 Recent Results")
    if recent_results_df is None or recent_results_df.empty:
        st.info("No recent Atlanta United results available.")
    else:
        recent_display = recent_results_df.copy()
        recent_display["Date (UTC)"] = recent_display["match_kickoff"].apply(
            lambda dt: dt.strftime("%Y-%m-%d %H:%M UTC") if pd.notna(dt) else "TBD"
        )
        recent_display["Result"] = recent_display.apply(
            lambda row: f"{row['home_team']} {int(row['final_home'])} - {int(row['final_away'])} {row['away_team']}",
            axis=1,
        )
        st.dataframe(recent_display[["Date (UTC)", "Result"]], use_container_width=True, hide_index=True)

with matches_tab:
    st.markdown(f"### {MLS_SEASON} Atlanta United Full Schedule")
    if schedule_df.empty:
        st.info("No schedule entries found in schedule.json.")
    else:
        matches_display = schedule_df.copy()

        score_map: dict[str, tuple[Optional[int], Optional[int]]] = {}
        if not predictions_df.empty and "match_id" in predictions_df.columns:
            scored = predictions_df.copy()
            for col in ["final_home", "final_away"]:
                if col not in scored.columns:
                    scored[col] = pd.NA
                scored[col] = pd.to_numeric(scored[col], errors="coerce")

            scored = scored[scored["final_home"].notna() & scored["final_away"].notna()].copy()
            scored = scored.sort_values("submitted_at") if "submitted_at" in scored.columns else scored
            for _, row in scored.iterrows():
                score_map[str(row["match_id"])] = (int(row["final_home"]), int(row["final_away"]))

        matches_display["final_home"] = matches_display["match_id"].astype(str).map(
            lambda mid: score_map.get(mid, (None, None))[0]
        )
        matches_display["final_away"] = matches_display["match_id"].astype(str).map(
            lambda mid: score_map.get(mid, (None, None))[1]
        )

        def schedule_status(row: pd.Series) -> str:
            if pd.notna(row.get("final_home")) and pd.notna(row.get("final_away")):
                return f"{row['home_team']} {int(row['final_home'])} - {int(row['final_away'])} {row['away_team']}"
            kickoff = row.get("match_kickoff")
            if pd.notna(kickoff) and pd.Timestamp(kickoff).to_pydatetime() <= now_utc:
                return "Match completed (score pending)"
            return "Scheduled"

        matches_display["Status"] = matches_display.apply(schedule_status, axis=1)
        matches_display["segment_label"] = matches_display["segment"].apply(
            lambda value: f"Segment {int(value)}" if pd.notna(value) else "TBD"
        )
        st.dataframe(
            matches_display[["segment_label", "kickoff_et", "home_team", "away_team", "Status"]]
            .rename(
                columns={
                    "segment_label": "Segment",
                    "kickoff_et": "Kickoff (ET)",
                    "home_team": "Home Team",
                    "away_team": "Away Team",
                }
            )
            .reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )

with leaderboard_tab:
    if active_segment_label and not active_segment_standings.empty:
        top_user = active_segment_standings.iloc[0]
        bottom_user = active_segment_standings.iloc[-1]
        st.markdown(
            f"### {active_segment_label}: 🏆 Gnomore Lossus — **{top_user['User']} ({int(top_user['Segment Points'])} pts)**"
        )
        st.markdown(
            f"### {active_segment_label}: 🥄 Wooden Spoon — **{bottom_user['User']} ({int(bottom_user['Segment Points'])} pts)**"
        )
    else:
        st.info("Segment trophies will appear once the active segment has completed matches.")

    st.divider()
    st.markdown("### Season Standings")
    if standings.empty:
        st.info("No prediction data available for leaderboard standings yet.")
    else:
        st.dataframe(standings[["Rank", "User", "Total Points"]], use_container_width=True, hide_index=True)

    segment_table_header = "### Segment Standings"
    if active_segment_label:
        segment_table_header = f"### Segment Standings ({active_segment_label})"
    st.markdown(segment_table_header)
    if active_segment_standings.empty:
        st.info("No completed-match points are available for the active segment yet.")
    else:
        st.dataframe(
            active_segment_standings[["Rank", "User", "Segment Points"]],
            use_container_width=True,
            hide_index=True,
        )

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
        existing_home = 0
        existing_away = 0
        if not predictions_df.empty and {"user_name", "match_id", "pred_home", "pred_away"}.issubset(predictions_df.columns):
            existing_prediction_mask = (
                predictions_df["user_name"].astype(str) == str(active_user_name)
            ) & (predictions_df["match_id"].astype(str) == str(next_match["match_id"]))
            existing_prediction = predictions_df.loc[existing_prediction_mask]
            if not existing_prediction.empty:
                existing_home_value = pd.to_numeric(existing_prediction.iloc[-1].get("pred_home"), errors="coerce")
                existing_away_value = pd.to_numeric(existing_prediction.iloc[-1].get("pred_away"), errors="coerce")
                if pd.notna(existing_home_value):
                    existing_home = int(existing_home_value)
                if pd.notna(existing_away_value):
                    existing_away = int(existing_away_value)

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
                value=existing_home,
                key=f"home_{next_match['match_id']}",
            )
            pred_away = col2.number_input(
                f"{next_match['away_team']} goals",
                min_value=0,
                max_value=15,
                step=1,
                value=existing_away,
                key=f"away_{next_match['match_id']}",
            )

            submitted = st.form_submit_button("Save Prediction")

        if submitted:
            if now_utc > next_match["match_kickoff"] + timedelta(minutes=15):
                st.error("Predictions are closed for this match (15-minute grace period has passed).")
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
- 3 points: exact score (home and away both match)
- 2 points: correct result (W/D/L) + one score matches
- 1 point: correct result (W/D/L) only
- 0 points: incorrect result (W/D/L) + one score matches
- -1 point: incorrect result (W/D/L) and neither score matches""",
        language="text",
    )

with faq_tab:
    st.markdown("### ❓ FAQ")

    with st.expander("⚽ Scoring Rules", expanded=True):
        st.markdown(
            """
            We use a **Precision Scaling** system to reward the best gurus and penalize the wild guesses:

            - **3 Points (The Bullseye):** You got the exact score right (e.g., you predicted 2-1 and it ended 2-1).
            - **2 Points (The Close Call):** You got the winner right **and** you nailed one team's score (e.g., you predicted 2-0 and it ended 2-1).
            - **1 Point (The Result):** You correctly picked the winner or a draw, but both scores were off.
            - **0 Points (The Silver Lining):** You got the winner/loser wrong, but you at least got one team's score right.
            - **-1 Point (The Wooden Spoon Effort):** You got the winner wrong **and** both scores wrong. Ouch.
            """
        )

    with st.expander("🏆 Trophies & Segments"):
        st.markdown(
            """
            To keep things interesting, the season is divided into **segments**:

            - **Segment 1:** Matches 1 and 2.
            - **Following Segments:** Every 4 matches thereafter.

            At the end of each segment, we crown two champions:

            - **Gnomore Lossus Trophy:** For the person with the most points in that segment.
            - **Wooden Spoon Trophy:** For the person at the bottom of the segment leaderboard.

            **Note:** Everyone starts at 0 at the beginning of a new segment, but your season total keeps climbing.
            """
        )

    with st.expander("🕒 Prediction Deadlines"):
        st.markdown(
            """
            - **Lockout:** Predictions lock 15 minutes after kickoff.
            - **Next Game Window:** The prediction window for the next match opens the day after the current match ends.
            """
        )

    with st.expander("💾 Persistence"):
        st.markdown(
            """
            You can change your prediction as many times as you want before lockout.
            The app only keeps your **latest** saved guess for that match.
            """
        )
