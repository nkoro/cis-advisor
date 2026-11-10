"""Offline tests for the conversational advisor.

No network and no API key: the streaming model client is stubbed.
"""

from types import SimpleNamespace

import pytest

import main
from main import (
    MAJOR_CHOICES,
    MAJOR_FILES,
    PII_MESSAGE,
    contains_pii,
    load_catalog,
    match_major,
    respond,
    start_over,
)

START = dict(main.DEFAULT_PROFILE)
AT_ID = {"stage": "student_id", "name": "Jane", "student_id": "", "major": ""}
AT_MAJOR = {"stage": "major", "name": "Jane", "student_id": "T1", "major": ""}
IN_CHAT = {"stage": "chat", "name": "Jane", "student_id": "T1", "major": "Computer Science"}


def turn(message, history=None, profile=None):
    """Run one turn and return (history, profile) from the final yield."""
    *_, last = main.respond(message, history or [], profile or START)
    return last[0], last[1]


@pytest.fixture()
def fake_stream(monkeypatch):
    """Replace the model stream with canned deltas."""

    def _install(pieces, error=None):
        captured = {}

        def _gen(messages):
            captured["messages"] = messages
            for piece in pieces:
                yield piece
            if error:
                raise error

        monkeypatch.setattr(main, "stream_reply", _gen)
        monkeypatch.setattr(main, "client", object())
        return captured

    return _install


@pytest.fixture()
def telemetry(monkeypatch, tmp_path):
    path = tmp_path / "telemetry.csv"
    monkeypatch.setattr(main, "TELEMETRY_PATH", path)
    return path


# --- Catalog wiring ----------------------------------------------------------
def test_every_major_has_a_catalog_file():
    for major in MAJOR_CHOICES:
        assert (main.DATA_DIR / MAJOR_FILES[major]).is_file(), major


def test_dropdown_choices_are_unique():
    assert len(MAJOR_CHOICES) == len(set(MAJOR_CHOICES))


def test_load_catalog_is_cached():
    main._CATALOG_CACHE.clear()
    first = load_catalog("Computer Science")
    assert load_catalog("Computer Science") is first


# --- Major matching ----------------------------------------------------------
@pytest.mark.parametrize(
    "answer,expected",
    [
        ("1", "Information Technology"),
        ("6", "Computer and Mathematical Sciences Secondary Education"),
        ("computer science", "Computer Science"),
        ("Computer Science", "Computer Science"),
        ("COMPUTER SCIENCE", "Computer Science"),
        ("information systems", "Information Systems"),
        ("accelerated", "Accelerated BS MS Computer Science"),
        ("secondary education",
         "Computer and Mathematical Sciences Secondary Education"),
    ],
)
def test_match_major_resolves(answer, expected):
    assert match_major(answer) == expected


@pytest.mark.parametrize("answer", ["", "9", "0", "underwater basket weaving"])
def test_match_major_rejects(answer):
    assert match_major(answer) is None


# --- PII guard ---------------------------------------------------------------
@pytest.mark.parametrize(
    "value", ["123-45-6789", "my ssn is 123-45-6789", "4111 1111 1111 1111", "4111111111111111"]
)
def test_contains_pii_flags(value):
    assert contains_pii(value) is True


@pytest.mark.parametrize(
    "value", ["", "Jane Doe", "T00000001", "COSC 236", "1234-5678", "123456789", "4111 1111 1111 1112"]
)
def test_contains_pii_allows(value):
    assert contains_pii(value) is False


def test_pii_is_blocked_before_calling_the_model(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("model must not be called")

    monkeypatch.setattr(main, "stream_reply", _boom)
    history, profile = turn("123-45-6789", profile=IN_CHAT)
    assert history[-1]["content"] == PII_MESSAGE
    assert profile == IN_CHAT


# --- Intake state machine ----------------------------------------------------
def converse(messages, history=None, profile=None):
    """Feed several messages through the handler, threading state forward."""
    history = list(history or [])
    profile = dict(profile or START)
    for text in messages:
        history, profile = turn(text, history, profile)
    return history, profile


def test_name_step_advances_to_student_id():
    history, profile = turn("Jane Doe")
    assert profile["name"] == "Jane Doe"
    assert profile["stage"] == "student_id"
    assert "student ID" in history[-1]["content"]


def test_invalid_name_keeps_the_name_step():
    history, profile = turn("Jane123")
    assert profile["stage"] == "name"
    assert profile["name"] == ""
    assert "letters" in history[-1]["content"]


def test_student_id_step_shows_the_program_menu():
    history, profile = turn("t01234567", profile=AT_ID)
    assert profile["student_id"] == "T01234567"
    assert profile["stage"] == "major"
    for major in MAJOR_CHOICES:
        assert major in history[-1]["content"]


def test_invalid_student_id_keeps_the_step():
    history, profile = turn("x!", profile=AT_ID)
    assert profile["stage"] == "student_id"
    assert profile["student_id"] == ""


def test_major_by_number_opens_the_conversation():
    history, profile = turn("5", profile=AT_MAJOR)
    assert profile["stage"] == "chat"
    assert profile["major"] == "Computer Science"
    assert "What can I do for you today" in history[-1]["content"]


def test_unmatched_major_reasks():
    history, profile = turn("underwater basket weaving", profile=AT_MAJOR)
    assert profile["stage"] == "major"
    assert "match" in history[-1]["content"]


def test_full_intake_then_advising(fake_stream):
    captured = fake_stream(["Hello ", "Jane!"])
    history, profile = converse(
        ["Jane Doe", "T01234567", "computer science", "What should I take next?"]
    )
    assert profile["stage"] == "chat"
    assert profile["major"] == "Computer Science"
    assert history[-1]["content"] == "Hello Jane!"

    system = captured["messages"][0]
    assert system["role"] == "system"
    assert "Jane Doe" in system["content"]
    assert "CATALOG FOR Computer Science" in system["content"]


def test_system_prompt_enforces_advising_scope_and_prerequisites(fake_stream):
    captured = fake_stream(["ok"])
    turn("hi", profile=IN_CHAT)
    content = captured["messages"][0]["content"]
    assert "Only course and degree advising" in content
    assert "Never schedule a course the student has already" in content
    assert "Never invent a course" in content


def test_only_recent_history_is_resent(fake_stream):
    captured = fake_stream(["ok"])
    long_history = [{"role": "user", "content": f"m{i}"} for i in range(40)]
    turn("hi", history=long_history, profile=IN_CHAT)
    assert len(captured["messages"]) == 1 + main.HISTORY_TURNS


# --- Failure handling --------------------------------------------------------
def test_missing_api_key_is_reported(monkeypatch):
    monkeypatch.setattr(main, "client", None)
    history, _ = turn("hi", profile=IN_CHAT)
    assert "SILICONFLOW_API_KEY" in history[-1]["content"]


def test_model_errors_are_surfaced(fake_stream):
    fake_stream([], error=RuntimeError("boom"))
    history, _ = turn("hi", profile=IN_CHAT)
    assert "error" in history[-1]["content"].lower()


def test_empty_message_is_ignored(fake_stream):
    fake_stream(["should not run"])
    history, profile = turn("   ", profile=IN_CHAT)
    assert history == []
    assert profile == IN_CHAT


# --- Telemetry ---------------------------------------------------------------
def test_telemetry_row_written_on_success(fake_stream, telemetry):
    fake_stream(["hi"])
    turn("hello", profile=IN_CHAT)
    lines = telemetry.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "timestamp,latency_seconds,major"
    assert lines[1].endswith(",Computer Science")


def test_telemetry_skipped_on_failure(fake_stream, telemetry):
    fake_stream([], error=RuntimeError("x"))
    turn("hello", profile=IN_CHAT)
    assert not telemetry.exists()


def test_telemetry_failure_never_breaks_chat(fake_stream, monkeypatch, tmp_path):
    fake_stream(["hi"])
    monkeypatch.setattr(main, "TELEMETRY_PATH", tmp_path)
    history, _ = turn("hello", profile=IN_CHAT)
    assert history[-1]["content"] == "hi"


# --- Restart -----------------------------------------------------------------
def test_start_over_resets_everything():
    history, profile, box = start_over()
    assert profile == main.DEFAULT_PROFILE
    assert box == ""
    assert "What can I do for you today" not in history[0]["content"]
    assert "name" in history[0]["content"].lower()
