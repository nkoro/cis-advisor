"""CIS Degree Plan Advisor.

Gradio front end plus SiliconFlow (Qwen2.5-72B) routing logic for drafting
Towson University CIS degree completion plans from the local markdown catalog
in ``data/``.

All configuration is environment driven -- see ``.env.example``.
"""

import csv
import os
import re
import time
import warnings
from pathlib import Path
from typing import List

import gradio as gr
import httpx
import instructor
from openai import OpenAI
from pydantic import BaseModel

# --- Paths -------------------------------------------------------------------
# This file lives at <repo root>/app/main.py, so resolve data/ relative to this
# file rather than the current working directory. `python app/main.py`,
# `python -m app.main` and container runs then all behave identically.
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
TELEMETRY_PATH = DATA_DIR / "telemetry.csv"

# Optional convenience for local development; never required.
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    pass

# --- Configuration -----------------------------------------------------------
MODEL_NAME = os.environ.get("SILICONFLOW_MODEL", "Qwen/Qwen2.5-72B-Instruct")
API_BASE_URL = os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.com/v1")
API_KEY = os.environ.get("SILICONFLOW_API_KEY", "").strip()

# Campus networks sometimes perform TLS inspection with an untrusted root CA,
# which makes httpx reject the certificate. Opt in to insecure TLS explicitly
# instead of silently disabling verification for every request.
INSECURE_TLS = os.environ.get("SILICONFLOW_INSECURE_TLS", "").strip().lower() in {
    "1",
    "true",
    "yes",
}

if INSECURE_TLS:
    warnings.filterwarnings("ignore", message="Unverified HTTPS request")

# Explicit dropdown label -> catalog filename mapping. A missing entry is a hard
# error rather than a silently generated (and therefore wrong) filename.
MAJOR_FILES = {
    "Information Technology": "information_technology.md",
    "Computer and Mathematical Sciences": "computer_and_mathematical_sciences.md",
    "Information Systems": "information_systems.md",
    "Accelerated BS MS Computer Science": "accelerated_bs_ms_computer_science.md",
    "Computer Science": "computer_science.md",
    "Computer and Mathematical Sciences Secondary Education": (
        "computer_and_mathematical_sciences_secondary_education.md"
    ),
}
MAJOR_CHOICES = list(MAJOR_FILES)

PII_MESSAGE = "ERROR: Sensitive PII detected."

SYSTEM_PROMPT = """You are a strict academic advisor for Towson University.
Your sole purpose is to draft Degree Completion Plans based EXACTLY on the provided context.

STRICT RULES:
1. Do not answer questions outside of degree planning.
2. NEVER place a course the student has already taken (or is currently taking) into a
   future semester. Treat those courses as satisfied and omit them from the plan.
3. Verify prerequisites are met before placing a course in a future semester.
4. Use course codes and titles exactly as they appear in the catalog. Where the
   catalog itself uses generic labels ("Core 1" through "Core 14", "Elective"),
   keep that label, but never invent new placeholder tokens such as "Elective1"
   or "Course A".
5. List each course at most once across the entire plan.
6. Respect the student's stated schedule preferences.

CATALOG CONTEXT:
{catalog}
"""


# --- Structured output schema ------------------------------------------------
class SemesterPlan(BaseModel):
    semester_name: str
    courses: List[str]
    total_credits: int


class DegreePlanDraft(BaseModel):
    remaining_requirements_met: bool
    plan: List[SemesterPlan]
    advisor_notes: str


# --- PII guard ---------------------------------------------------------------
# Enforced deterministically in code: a prompt instruction alone cannot satisfy a
# strict JSON response schema, so the model cannot reliably emit the refusal text.
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")


def _luhn_ok(digits: str) -> bool:
    """Standard Luhn checksum, used to avoid flagging arbitrary long numbers."""
    total, parity = 0, len(digits) % 2
    for index, char in enumerate(digits):
        value = int(char)
        if index % 2 == parity:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def contains_pii(*values: str) -> bool:
    """True when any value looks like an SSN or a Luhn-valid card number."""
    text = " ".join(value or "" for value in values)
    if SSN_RE.search(text):
        return True
    for candidate in CARD_RE.findall(text):
        digits = re.sub(r"\D", "", candidate)
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            return True
    return False


# --- SiliconFlow client ------------------------------------------------------
def _build_client():
    """Build the Instructor client, or return None when no key is configured."""
    if not API_KEY:
        return None
    return instructor.from_openai(
        OpenAI(
            base_url=API_BASE_URL,
            api_key=API_KEY,
            http_client=httpx.Client(
                verify=not INSECURE_TLS,
                timeout=httpx.Timeout(
                    float(os.environ.get("SILICONFLOW_TIMEOUT", "180")), connect=15.0
                ),
            ),
            max_retries=2,
        ),
        mode=instructor.Mode.JSON,
    )


client = _build_client()


# --- Telemetry ---------------------------------------------------------------
def _log_telemetry(latency: float, major: str) -> None:
    """Append one research row. Telemetry must never break a plan."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        is_new = not TELEMETRY_PATH.exists()
        with TELEMETRY_PATH.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if is_new:
                writer.writerow(["timestamp", "latency_seconds", "major"])
            writer.writerow([round(time.time(), 3), latency, major])
    except OSError:
        pass


# --- Output formatting -------------------------------------------------------
def _format_plan(name: str, student_id: str, resp: DegreePlanDraft, latency: float) -> str:
    total = sum(semester.total_credits for semester in resp.plan)
    status = "On Track" if resp.remaining_requirements_met else "Outstanding Requirements Remain"
    output = f"### Plan for {name} ({student_id})\n**Status:** {status}\n\n"
    for semester in resp.plan:
        output += f"**{semester.semester_name}** ({semester.total_credits} credits)\n"
        for course in semester.courses:
            output += f"- {course}\n"
        output += "\n"
    output += f"**Total planned credits:** {total}\n\n"
    output += f"**Advisor Notes:**\n{resp.advisor_notes}\n\n"
    output += f"*Generated in {latency} seconds using SiliconFlow ({MODEL_NAME})*"
    return output


# --- Core advisory function --------------------------------------------------
def generate_plan(name, student_id, major, courses_taken, preferences):
    start_time = time.time()

    if client is None:
        return (
            "**Configuration error:** `SILICONFLOW_API_KEY` is not set. "
            "Add it to your environment (or a local `.env` file) and restart the app."
        )

    if not major or major not in MAJOR_FILES:
        return "Error: please select a major from the list."

    if contains_pii(name, student_id, courses_taken, preferences):
        return PII_MESSAGE

    file_path = DATA_DIR / MAJOR_FILES[major]
    try:
        major_context = file_path.read_text(encoding="utf-8")
    except OSError:
        return f"Error: context file for {major} not found ({file_path})."

    user_prompt = (
        f"Student: {name} ({student_id})\n"
        f"Already taken / currently taking (do NOT reschedule these): "
        f"{courses_taken.strip() or 'none reported'}\n"
        f"Preferences: {preferences.strip() or 'none'}"
    )

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            response_model=DegreePlanDraft,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.format(catalog=major_context)},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,  # Keep it deterministic
        )
    except Exception as exc:  # noqa: BLE001 - surfaced directly in the UI
        return f"Inference Error: {exc}"

    latency = round(time.time() - start_time, 2)
    _log_telemetry(latency, major)
    return _format_plan(name, student_id, resp, latency)


# --- Gradio UI ---------------------------------------------------------------
def _launch_auth():
    user = os.environ.get("ADVISOR_USERNAME", "").strip()
    password = os.environ.get("ADVISOR_PASSWORD", "").strip()
    if user and password:
        return (user, password)
    print("[warn] ADVISOR_USERNAME/ADVISOR_PASSWORD not set - launching with no login.")
    return None


with gr.Blocks(title="CIS Degree Plan Advisor") as ui:
    gr.Markdown("# CIS Degree Plan Advisor")
    gr.Markdown(
        "Drafts a Towson University degree completion plan from the official catalog "
        f"context. Plans are generated by `{MODEL_NAME}` via the SiliconFlow API."
    )
    with gr.Row():
        with gr.Column():
            name = gr.Textbox(label="Student Name")
            student_id = gr.Textbox(label="Student ID")
            major = gr.Dropdown(choices=MAJOR_CHOICES, label="Select Major")
            courses = gr.Textbox(label="Courses Taken/Taking (Comma separated)", lines=3)
            prefs = gr.Textbox(
                label="Schedule Preferences (e.g., Max 15 credits, no mornings)", lines=2
            )
            submit = gr.Button("Generate Plan", variant="primary")

        with gr.Column():
            output_display = gr.Markdown(label="Generated Plan")

    submit.click(
        fn=generate_plan,
        inputs=[name, student_id, major, courses, prefs],
        outputs=output_display,
    )


if __name__ == "__main__":
    ui.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        auth=_launch_auth(),
    )

