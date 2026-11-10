"""CIS Degree Plan Advisor - a conversational academic advisor.

Gradio chat UI backed by the SiliconFlow API, grounded in a local markdown copy
of the Towson University catalog (``data/*.md``).

Configuration is environment driven -- see ``.env.example``.
"""

import csv
import os
import re
import time
import warnings
from pathlib import Path
from typing import Iterator, Optional

import gradio as gr
import httpx
from openai import OpenAI

# --- Paths -------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
TELEMETRY_PATH = DATA_DIR / "telemetry.csv"

try:  # optional local convenience
    from dotenv import load_dotenv

    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    pass

# --- Configuration -----------------------------------------------------------
# Chosen by benchmark (Nov 2026): fastest of the cheap tier (~1.3s/turn) and, via
# SiliconFlow prefix caching, ~99% of the ~19k-token catalog is billed at the
# cached rate ($0.003/M) instead of $0.15/M. Measured ~$0.00013 per turn.
MODEL_NAME = os.environ.get("SILICONFLOW_MODEL", "deepseek-ai/DeepSeek-V4.1-Flash")
API_BASE_URL = os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.com/v1")
API_KEY = os.environ.get("SILICONFLOW_API_KEY", "").strip()
MAX_TOKENS = int(os.environ.get("SILICONFLOW_MAX_TOKENS", "900"))
HISTORY_TURNS = int(os.environ.get("ADVISOR_HISTORY_TURNS", "12"))

INSECURE_TLS = os.environ.get("SILICONFLOW_INSECURE_TLS", "").strip().lower() in {"1", "true", "yes"}
if INSECURE_TLS:
    warnings.filterwarnings("ignore", message="Unverified HTTPS request")

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
MAJOR_MENU = "\n".join(f"{i}. {name}" for i, name in enumerate(MAJOR_CHOICES, 1))

DEFAULT_PROFILE = {"stage": "name", "name": "", "student_id": "", "major": ""}

# --- PII guard ---------------------------------------------------------------
# Enforced in code, before anything reaches the API.
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
NAME_RE = re.compile(r"^[A-Za-z][A-Za-z .'\-]{0,49}$")
STUDENT_ID_RE = re.compile(r"^[A-Za-z0-9\-]{3,20}$")

PII_MESSAGE = (
    "I'd rather not handle that here - it looks like a Social Security or card "
    "number. Please keep those to yourself. What can I help you with regarding "
    "your degree?"
)


def _luhn_ok(digits: str) -> bool:
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


# --- Conversation copy -------------------------------------------------------
WELCOME = (
    "\U0001F44B Hi! I'm the **CIS Advisor** for Towson University's Department of "
    "Computer and Information Sciences.\n\nBefore we start, **what's your name?**"
)

ASK_STUDENT_ID = "Great to meet you, **{name}**! \U0001F393\n\nWhat's your **student ID**?"

ASK_MAJOR = "Thanks! Which program are you in? Reply with a number or the name:\n\n{menu}"

GREETING = (
    "Perfect - **{major}**. I've got that catalog in front of me, **{name}**.\n\n"
    "**What can I do for you today?** For example:\n"
    "- *What should I take next semester?*\n"
    "- *Am I on track to graduate?*\n"
    "- *Build me a plan for my remaining semesters.*"
)


# --- Advisor prompt ----------------------------------------------------------
# NOTE: the catalog is appended with plain concatenation, never str.format(),
# because catalog text can contain braces.
ADVISOR_SYSTEM = """You are "CIS Advisor", a warm, professional academic advisor in Towson University's Department of Computer and Information Sciences.

STUDENT ON FILE
- Name: {name}
- Student ID: {student_id}
- Program: {major}

WHAT YOU HELP WITH
Only course and degree advising for this student: degree requirements, which courses to take and in what order, prerequisites, electives, credit loads, graduation progress and scheduling. If the student asks about anything else (other majors, campus admin, personal matters, general chit-chat, coding help), decline in one short friendly sentence and steer back to their degree plan.

HOW TO CONVERSE
- Warm and concise: 2-5 sentences unless you are presenting a plan.
- Ask at most two questions per message; never fire a long questionnaire.
- When a planning request first comes up, work out naturally over a message or two: (1) which courses they have completed or are taking now, (2) which terms the plan should cover, (3) preferences such as credits per semester, mornings or evenings, online or on campus. Skip anything they have already told you and never re-ask it.
- When you have enough, present the plan as a markdown table: Term | Course | Title | Credits, then the total credits.

ACCURACY RULES
- Use ONLY the catalog below for requirements, prerequisites and credit values. If something is not there, say you cannot confirm it and suggest checking with the department. Never invent a course, prerequisite or credit value.
- Never schedule a course the student has already completed or is currently taking.
- Confirm a prerequisite is satisfied before placing a course in a later term.
- The catalog's own generic labels ("Core 1" to "Core 14", "Elective") are legitimate names; reuse them exactly and do not invent new placeholder codes."""

CONFIG_ERROR = (
    "**Configuration error:** `SILICONFLOW_API_KEY` is not set. Add it to your "
    "environment or `.env` file and restart the app."
)

_CATALOG_CACHE = {}


def load_catalog(major: str) -> str:
    """Read a major's catalog once and keep it in memory."""
    if major not in _CATALOG_CACHE:
        _CATALOG_CACHE[major] = (DATA_DIR / MAJOR_FILES[major]).read_text(encoding="utf-8")
    return _CATALOG_CACHE[major]


def advisor_system(profile, catalog: str) -> str:
    header = ADVISOR_SYSTEM.format(
        name=profile.get("name") or "Unknown",
        student_id=profile.get("student_id") or "Unknown",
        major=profile.get("major") or "Unknown",
    )
    return header + "\n\nCATALOG FOR " + str(profile.get("major")) + "\n" + catalog


def _normalise(text: str) -> str:
    text = (text or "").lower().replace("&", " and ")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def match_major(answer: str) -> Optional[str]:
    """Resolve a typed answer to a program name (number, exact or partial)."""
    text = (answer or "").strip()
    if not text:
        return None
    if text.isdigit():
        index = int(text)
        return MAJOR_CHOICES[index - 1] if 1 <= index <= len(MAJOR_CHOICES) else None
    target = _normalise(text)
    if not target:
        return None
    for major in MAJOR_CHOICES:
        if _normalise(major) == target:
            return major
    partial = [
        major
        for major in MAJOR_CHOICES
        if _normalise(major) in target or target in _normalise(major)
    ]
    return max(partial, key=lambda major: len(_normalise(major))) if partial else None


def _build_client():
    if not API_KEY:
        return None
    return OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
        http_client=httpx.Client(
            verify=not INSECURE_TLS,
            timeout=httpx.Timeout(float(os.environ.get("SILICONFLOW_TIMEOUT", "180")), connect=15.0),
        ),
        max_retries=2,
    )


client = _build_client()

def stream_reply(messages) -> Iterator[str]:
    """Yield text deltas from the model."""
    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.3,
        max_tokens=MAX_TOKENS,
        stream=True,
    )
    for chunk in stream:
        choices = getattr(chunk, "choices", None)
        if not choices:
            continue
        piece = getattr(choices[0].delta, "content", None)
        if piece:
            yield piece


def log_telemetry(latency: float, major: str) -> None:
    """Append one research row. Telemetry must never break a conversation."""
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


# --- Chat handler ------------------------------------------------------------
def respond(message, history, profile):
    """One chat turn: run the intake state machine, then stream advising."""
    profile = {**DEFAULT_PROFILE, **(profile or {})}
    history = list(history or [])
    text = (message or "").strip()

    if not text:
        yield history, profile, ""
        return

    history.append({"role": "user", "content": text})

    if contains_pii(text):
        history.append({"role": "assistant", "content": PII_MESSAGE})
        yield history, profile, ""
        return

    stage = profile["stage"]

    # --- intake ---------------------------------------------------------
    if stage == "name":
        if not NAME_RE.match(text):
            history.append(
                {
                    "role": "assistant",
                    "content": "Sorry, I need a name to get started - letters only, please.",
                }
            )
        else:
            profile["name"] = " ".join(text.split())
            profile["stage"] = "student_id"
            history.append(
                {
                    "role": "assistant",
                    "content": ASK_STUDENT_ID.format(name=profile["name"]),
                }
            )
        yield history, profile, ""
        return

    if stage == "student_id":
        if not STUDENT_ID_RE.match(text):
            history.append(
                {
                    "role": "assistant",
                    "content": "That doesn't look like a student ID. Send it as letters "
                    "and numbers, e.g. `T01234567`.",
                }
            )
        else:
            profile["student_id"] = text.upper()
            profile["stage"] = "major"
            history.append({"role": "assistant", "content": ASK_MAJOR.format(menu=MAJOR_MENU)})
        yield history, profile, ""
        return

    if stage == "major":
        major = match_major(text)
        if major is None:
            history.append(
                {
                    "role": "assistant",
                    "content": "I couldn't match that to a program.\n\n"
                    + ASK_MAJOR.format(menu=MAJOR_MENU),
                }
            )
        else:
            profile["major"] = major
            profile["stage"] = "chat"
            history.append(
                {
                    "role": "assistant",
                    "content": GREETING.format(name=profile["name"], major=major),
                }
            )
        yield history, profile, ""
        return

    # --- advising -------------------------------------------------------
    if client is None or not profile.get("major"):
        history.append({"role": "assistant", "content": CONFIG_ERROR})
        yield history, profile, ""
        return

    try:
        catalog = load_catalog(profile["major"])
    except OSError:
        history.append(
            {
                "role": "assistant",
                "content": f"Sorry, I can't load the catalog for {profile['major']}.",
            }
        )
        yield history, profile, ""
        return

    messages = [{"role": "system", "content": advisor_system(profile, catalog)}]
    messages += history[-HISTORY_TURNS:]

    started = time.time()
    answer = ""
    failed = False
    history.append({"role": "assistant", "content": "..."})
    try:
        for piece in stream_reply(messages):
            answer += piece
            yield history[:-1] + [{"role": "assistant", "content": answer}], profile, ""
    except Exception as exc:  # noqa: BLE001 - surfaced to the student
        failed = True
        answer = answer or f"Sorry - I hit an error reaching the model.\n\n`{str(exc)[:300]}`"

    history[-1] = {"role": "assistant", "content": answer or "_(no response generated)_"}
    if not failed and answer:
        log_telemetry(round(time.time() - started, 2), profile["major"])
    yield history, profile, ""


def start_over():
    return [{"role": "assistant", "content": WELCOME}], dict(DEFAULT_PROFILE), ""


def _launch_auth():
    user = os.environ.get("ADVISOR_USERNAME", "").strip()
    password = os.environ.get("ADVISOR_PASSWORD", "").strip()
    if user and password:
        return (user, password)
    print("[warn] ADVISOR_USERNAME/ADVISOR_PASSWORD not set - launching with no login.")
    return None


# --- UI ----------------------------------------------------------------------
with gr.Blocks(title="CIS Degree Plan Advisor") as ui:
    gr.Markdown("# \U0001F393 CIS Degree Plan Advisor")
    gr.Markdown(
        "A conversational academic advisor for Towson University's Department of "
        "Computer and Information Sciences. Every answer is grounded in the "
        f"official catalog, via `{MODEL_NAME}`."
    )

    chatbot = gr.Chatbot(
        value=[{"role": "assistant", "content": WELCOME}],
        height=520,
        label="CIS Advisor",
        line_breaks=True,
    )
    with gr.Row():
        message = gr.Textbox(
            placeholder="Type your answer or question...",
            show_label=False,
            scale=8,
            lines=1,
        )
        send = gr.Button("Send", variant="primary", scale=1)
    reset = gr.Button("Start over", variant="secondary")

    profile = gr.State(dict(DEFAULT_PROFILE))

    send.click(respond, [message, chatbot, profile], [chatbot, profile, message])
    message.submit(respond, [message, chatbot, profile], [chatbot, profile, message])
    reset.click(start_over, None, [chatbot, profile, message])


if __name__ == "__main__":
    ui.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        auth=_launch_auth(),
    )
