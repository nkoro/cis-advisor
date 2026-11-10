"""Offline tests for the advisor's deterministic logic.

No network access and no API key are required: the SiliconFlow/Instructor client
is replaced with a stub, so these tests are safe to run in CI.
"""

import pytest

import main
from main import (
    MAJOR_CHOICES,
    MAJOR_FILES,
    DegreePlanDraft,
    SemesterPlan,
    contains_pii,
    generate_plan,
)


class FakeCompletions:
    """Records what the app sends to the model and replays a canned reply."""

    def __init__(self, reply=None, error=None):
        self.reply = reply
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.reply


class FakeClient:
    def __init__(self, reply=None, error=None):
        self.chat = type("Chat", (), {})()
        self.chat.completions = FakeCompletions(reply, error)


def make_reply(*, met=True, notes="All good."):
    return DegreePlanDraft(
        remaining_requirements_met=met,
        plan=[
            SemesterPlan(
                semester_name="Fall 2026",
                courses=["COSC 336", "ITEC 315"],
                total_credits=6,
            ),
            SemesterPlan(semester_name="Spring 2027", courses=["COSC 418"], total_credits=3),
        ],
        advisor_notes=notes,
    )


@pytest.fixture()
def stub_client(monkeypatch):
    """Install a fake client so no test ever reaches the network."""

    def _install(reply=None, error=None):
        fake = FakeClient(reply if reply is not None else make_reply(), error)
        monkeypatch.setattr(main, "client", fake)
        return fake

    return _install


@pytest.fixture()
def temp_telemetry(monkeypatch, tmp_path):
    path = tmp_path / "telemetry.csv"
    monkeypatch.setattr(main, "TELEMETRY_PATH", path)
    return path


# --- Catalog wiring ----------------------------------------------------------
def test_every_dropdown_major_maps_to_an_existing_file():
    for label in MAJOR_CHOICES:
        assert (main.DATA_DIR / MAJOR_FILES[label]).is_file(), f"missing catalog for {label}"


def test_dropdown_choices_are_unique():
    assert len(MAJOR_CHOICES) == len(set(MAJOR_CHOICES))


# --- PII guard ---------------------------------------------------------------
@pytest.mark.parametrize(
    "value",
    [
        "123-45-6789",
        "my ssn is 123-45-6789 ok",
        "card 4111 1111 1111 1111",
        "4111111111111111",
    ],
)
def test_contains_pii_flags_sensitive_values(value):
    assert contains_pii(value) is True


@pytest.mark.parametrize(
    "value",
    [
        "",
        "Jane Doe",
        "T00000001",
        "COSC 236, COSC 237",
        "no mornings",
        "1234-5678",
        "123456789",
        "4111 1111 1111 1112",  # right length, fails the Luhn check
    ],
)
def test_contains_pii_allows_normal_values(value):
    assert contains_pii(value) is False


# --- Guard rails -------------------------------------------------------------
def test_pii_is_blocked_before_the_model_is_called(stub_client):
    fake = stub_client()
    out = generate_plan("Jane", "123-45-6789", "Computer Science", "COSC 236", "")
    assert out == main.PII_MESSAGE
    assert fake.chat.completions.calls == []


def test_missing_api_key_is_reported(monkeypatch):
    monkeypatch.setattr(main, "client", None)
    out = generate_plan("Jane", "T1", "Computer Science", "", "")
    assert "SILICONFLOW_API_KEY" in out


def test_unknown_major_is_rejected(stub_client):
    stub_client()
    out = generate_plan("Jane", "T1", "Underwater Basket Weaving", "", "")
    assert out.startswith("Error:")


# --- Happy path --------------------------------------------------------------
def test_generate_plan_formats_the_plan(stub_client, temp_telemetry):
    stub_client()
    out = generate_plan("Jane Doe", "T00000001", "Computer Science", "COSC 236", "Max 15 credits")
    assert "### Plan for Jane Doe (T00000001)" in out
    assert "**Fall 2026** (6 credits)" in out
    assert "- COSC 336" in out
    assert "- COSC 418" in out
    assert "**Total planned credits:** 9" in out
    assert "All good." in out
    assert "SiliconFlow" in out


def test_taken_courses_model_and_catalog_reach_the_api(stub_client, temp_telemetry):
    fake = stub_client()
    generate_plan(
        "Jane Doe", "T00000001", "Computer Science", "COSC 236, COSC 237", "no mornings"
    )
    call = fake.chat.completions.calls[0]
    assert call["model"] == main.MODEL_NAME
    assert call["temperature"] == 0.1
    assert call["response_model"] is DegreePlanDraft

    system_message = call["messages"][0]["content"]
    user_message = call["messages"][1]["content"]
    assert "{catalog}" not in system_message
    assert "Towson" in system_message
    assert "do NOT reschedule" in user_message
    assert "COSC 236, COSC 237" in user_message
    assert "no mornings" in user_message


def test_status_label_reflects_remaining_requirements(stub_client, temp_telemetry):
    stub_client(reply=make_reply(met=True))
    assert "On Track" in generate_plan("J", "T1", "Computer Science", "", "")

    stub_client(reply=make_reply(met=False))
    assert "Outstanding Requirements Remain" in generate_plan("J", "T1", "Computer Science", "", "")


# --- Telemetry ---------------------------------------------------------------
def test_telemetry_row_is_written_with_header(stub_client, temp_telemetry):
    stub_client()
    generate_plan("Jane Doe", "T00000001", "Computer Science", "", "")
    lines = temp_telemetry.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "timestamp,latency_seconds,major"
    assert lines[1].endswith(",Computer Science")
    assert len(lines) == 2


def test_second_run_appends_without_duplicating_the_header(stub_client, temp_telemetry):
    stub_client()
    generate_plan("Jane", "T1", "Computer Science", "", "")
    generate_plan("Jane", "T1", "Computer Science", "", "")
    lines = temp_telemetry.read_text(encoding="utf-8").splitlines()
    assert lines.count("timestamp,latency_seconds,major") == 1
    assert len(lines) == 3


def test_telemetry_write_failure_never_breaks_the_plan(monkeypatch, stub_client, tmp_path):
    stub_client()
    # A directory is not appendable, so the write raises OSError internally.
    monkeypatch.setattr(main, "TELEMETRY_PATH", tmp_path)
    assert "### Plan for Jane" in generate_plan("Jane", "T1", "Computer Science", "", "")


# --- Error surfacing ---------------------------------------------------------
def test_inference_errors_are_surfaced_without_touching_telemetry(stub_client, temp_telemetry):
    stub_client(error=RuntimeError("boom"))
    out = generate_plan("Jane", "T1", "Computer Science", "", "")
    assert out == "Inference Error: boom"
    assert not temp_telemetry.exists()
