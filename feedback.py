"""
feedback.py - usage logging, in-app feedback form and admin export for X-Learn AI.

Produces the user-testing evidence for the Design Thinking Lab rubric:
  * events.csv   - what testers actually did (predictions, quiz answers, scores)
  * feedback.csv - structured questionnaire responses (1-5 ratings + comments)
  * admin_panel  - password-protected summary + CSV download (open the app with ?admin=1)

Set ADMIN_PASSWORD in .streamlit/secrets.toml (or Streamlit Cloud secrets) and
open the app with ?admin=1 to see the admin panel. Logging never raises, so it can't break the app.
"""
import csv
import json
import os
import threading
import uuid
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

LOG_DIR = os.environ.get("XLEARN_LOG_DIR", "logs")
EVENTS_CSV = os.path.join(LOG_DIR, "events.csv")
FEEDBACK_CSV = os.path.join(LOG_DIR, "feedback.csv")

EVENT_COLS = ["timestamp", "session_id", "event", "detail"]
FEEDBACK_COLS = [
    "timestamp", "session_id", "tester_role", "ease_of_use",
    "learning_value", "heatmap_clarity", "confusing", "improve",
]

_lock = threading.Lock()


def session_id():
    if "xl_sid" not in st.session_state:
        st.session_state["xl_sid"] = uuid.uuid4().hex[:8]
    return st.session_state["xl_sid"]


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _append(path, cols, row):
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with _lock:
            is_new = not os.path.exists(path)
            with open(path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=cols)
                if is_new:
                    writer.writeheader()
                writer.writerow(row)
    except Exception:
        pass  # never break the app because of logging


def log_event(event, **detail):
    """e.g. log_event("quiz_answer", difficulty="Easy", correct=True, ai_agreed=True)"""
    _append(EVENTS_CSV, EVENT_COLS, {
        "timestamp": _now(),
        "session_id": session_id(),
        "event": event,
        "detail": json.dumps(detail, default=str),
    })


def feedback_form():
    """Anonymous questionnaire. Call it on the quiz results screen / sidebar."""
    if st.session_state.get("xl_fb_done"):
        st.success("Thanks - your feedback was recorded.")
        return

    st.markdown("### Rate your experience")
    st.caption("Anonymous - takes about a minute.")
    with st.form("xl_feedback_form"):
        role = st.selectbox(
            "I am a...",
            ["Medical / allied health student", "Engineering student",
             "Doctor / radiographer / faculty", "Other"],
            index=None, placeholder="Select one",
        )
        ease = st.radio("The app was easy to use (1 = very hard, 5 = very easy)",
                        [1, 2, 3, 4, 5], horizontal=True, index=None)
        learn = st.radio("The app helped me learn to spot fractures (1 = not at all, 5 = a lot)",
                         [1, 2, 3, 4, 5], horizontal=True, index=None)
        heat = st.radio("The highlighted regions helped me understand the AI's result",
                        ["N/A", 1, 2, 3, 4, 5], horizontal=True, index=None)
        confusing = st.text_area("What was confusing or frustrating? (optional)")
        improve = st.text_area("What one thing would you improve? (optional)")
        submitted = st.form_submit_button("Submit feedback")

    if submitted:
        if None in (role, ease, learn, heat):
            st.warning("Please answer the role and all three rating questions.")
            return
        _append(FEEDBACK_CSV, FEEDBACK_COLS, {
            "timestamp": _now(), "session_id": session_id(),
            "tester_role": role, "ease_of_use": ease, "learning_value": learn,
            "heatmap_clarity": heat, "confusing": confusing.strip(),
            "improve": improve.strip(),
        })
        log_event("feedback_submitted")
        st.session_state["xl_fb_done"] = True
        st.rerun()


def _read(path, cols):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=cols)


def admin_panel():
    """Team-only summary + CSV export. Visible only at <app-url>/?admin=1 and with
    ADMIN_PASSWORD set in Streamlit secrets, so testers never see it."""
    try:
        if st.query_params.get("admin") != "1":
            return
        secret = st.secrets.get("ADMIN_PASSWORD")
    except Exception:
        return
    if not secret:
        return

    with st.expander("Admin"):
        if st.text_input("Password", type="password", key="xl_admin_pw") != secret:
            return

        ev = _read(EVENTS_CSV, EVENT_COLS)
        fb = _read(FEEDBACK_CSV, FEEDBACK_COLS)

        st.metric("Sessions logged", ev["session_id"].nunique() if not ev.empty else 0)
        st.metric("Feedback responses", len(fb))

        if not fb.empty:
            for col, label in [("ease_of_use", "Ease of use"),
                               ("learning_value", "Learning value"),
                               ("heatmap_clarity", "Heatmap clarity")]:
                avg = pd.to_numeric(fb[col], errors="coerce").mean()
                if pd.notna(avg):
                    st.metric(label + " (avg /5)", f"{avg:.2f}")

        qa = ev[ev["event"] == "quiz_answer"] if not ev.empty else ev
        if not qa.empty:
            d = pd.DataFrame(qa["detail"].apply(json.loads).tolist())
            if "correct" in d:
                st.metric("Quiz accuracy", f"{d['correct'].astype(float).mean():.0%}")
            if "ai_agreed" in d:
                st.metric("Student-AI agreement", f"{d['ai_agreed'].astype(float).mean():.0%}")

        st.download_button("Download events.csv", ev.to_csv(index=False).encode(),
                           "events.csv", "text/csv")
        st.download_button("Download feedback.csv", fb.to_csv(index=False).encode(),
                           "feedback.csv", "text/csv")
        if not fb.empty:
            st.dataframe(fb.drop(columns=["session_id"]), use_container_width=True)
