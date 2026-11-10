"""TUAdvisor - a conversational academic advisor.

Gradio chat UI backed by the SiliconFlow API, grounded in a local markdown copy
of the Towson University catalog (``data/*.md``).

Configuration is environment driven -- see ``.env.example``.
"""

import csv
import html
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
MAX_TOKENS = int(os.environ.get("SILICONFLOW_MAX_TOKENS", "1500"))
HISTORY_TURNS = int(os.environ.get("ADVISOR_HISTORY_TURNS", "12"))

# Reasoning-capable models bill hidden reasoning against max_tokens. Observed on
# this model: 900/900 tokens spent on reasoning and ZERO characters of visible
# reply, which surfaced to the student as an empty message. "none" disables it;
# set "low"/"medium"/"high" to trade latency and cost for deeper planning.
REASONING_EFFORT = os.environ.get("SILICONFLOW_REASONING_EFFORT", "none").strip().lower()

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
    "\U0001F44B Hi! I'm the **TUAdvisor** for Towson University's Department of "
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

def _stream_kwargs(max_tokens: Optional[int] = None) -> dict:
    """Request parameters for one model call."""
    kwargs = {
        "model": MODEL_NAME,
        "temperature": 0.3,
        "max_tokens": max_tokens or MAX_TOKENS,
        "stream": True,
    }
    if REASONING_EFFORT:
        kwargs["reasoning_effort"] = REASONING_EFFORT
    return kwargs


def stream_reply(messages, max_tokens: Optional[int] = None) -> Iterator[str]:
    """Yield visible text deltas from the model.

    ``reasoning_effort`` is passed only when configured, and a model that rejects
    it is retried without the parameter rather than failing the turn.
    """
    kwargs = _stream_kwargs(max_tokens)
    try:
        stream = client.chat.completions.create(messages=messages, **kwargs)
    except Exception:
        if "reasoning_effort" not in kwargs:
            raise
        kwargs.pop("reasoning_effort")
        stream = client.chat.completions.create(messages=messages, **kwargs)

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


# --- Theme -------------------------------------------------------------------
TU_GOLD = gr.themes.Color(
    name="tu_gold",
    c50="#FFF9E8",
    c100="#FFF1C6",
    c200="#FFE49B",
    c300="#FFD466",
    c400="#FFC53D",
    c500="#FFB81C",
    c600="#E0A100",
    c700="#B37F00",
    c800="#805B00",
    c900="#4D3700",
    c950="#261B00",
)
TU_INK = gr.themes.Color(
    name="tu_ink",
    c50="#F6F6F7",
    c100="#EDEDEF",
    c200="#DBDBDF",
    c300="#C2C2C8",
    c400="#9A9AA2",
    c500="#6E6E78",
    c600="#4B4B54",
    c700="#32323A",
    c800="#1F1F24",
    c900="#121216",
    c950="#000000",
)

# Blackboard-style neutrals with Towson black + gold; Open Sans is the closest
# widely-available match to the Learning/Edu UI stack.
THEME = gr.themes.Base(
    primary_hue=TU_GOLD,
    secondary_hue=TU_GOLD,
    neutral_hue=TU_INK,
    # Inter carries noticeably more weight than Open Sans at the same size, which
    # keeps the interface legible when a screenshot is scaled down for print.
    font=[gr.themes.GoogleFont("Inter"), "Inter", "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
)

CSS = """
.gradio-container { max-width: 1120px !important; margin: 0 auto !important; }

/* header */
#tu-header { display: flex; align-items: center; gap: 14px; margin: 10px 0 2px; }
#tu-header .tu-mark {
  flex: none; width: 46px; height: 46px; border-radius: 13px;
  background: #FFB81C; color: #111114; font-weight: 800; font-size: 17px;
  letter-spacing: -.5px; display: flex; align-items: center; justify-content: center;
}
#tu-header .tu-name { margin: 0; font-size: 26px; font-weight: 800; letter-spacing: -.6px; }
#tu-header .tu-tag { margin: 2px 0 0; font-size: 13.5px; color: var(--body-text-color-subdued); }

/* student card */
#tu-card {
  border: 1px solid var(--border-color-primary); border-radius: 14px;
  background: var(--block-background-fill); padding: 15px 17px;
}
#tu-card h4 {
  margin: 0 0 9px; font-size: 11px; letter-spacing: .09em; text-transform: uppercase;
  color: var(--body-text-color-subdued); font-weight: 700;
}
#tu-card .tu-row {
  display: flex; justify-content: space-between; gap: 12px; padding: 5px 0; font-size: 14px;
}
#tu-card .tu-k { color: var(--body-text-color-subdued); flex: none; }
#tu-card .tu-v { font-weight: 600; text-align: right; word-break: break-word; }
#tu-card .tu-hint, .tu-note {
  font-size: 12.5px; line-height: 1.55; color: var(--body-text-color-subdued);
}
.tu-note { padding: 10px 4px 0; }

/* chat surface */
#tu-chat { border-radius: 16px; }
#tu-chat p, #tu-chat li { line-height: 1.62; }

/* Heavier text so the interface stays readable when scaled into a printed page */
.gradio-container { font-weight: 500; }
#tu-chat p, #tu-chat li, #tu-chat td, #tu-chat th {
  font-weight: 500;
  font-size: 15.5px;
}
#tu-chat h1, #tu-chat h2, #tu-chat h3, #tu-chat strong { font-weight: 700; }
"""


def render_student_card(profile) -> str:
    """Small sidebar card showing whatever intake data we have so far."""
    profile = profile or {}
    rows = []
    if profile.get("name"):
        rows.append(("Student", profile["name"]))
    if profile.get("student_id"):
        rows.append(("ID", profile["student_id"]))
    if profile.get("major"):
        rows.append(("Program", profile["major"]))

    if rows:
        body = "".join(
            f'<div class="tu-row"><span class="tu-k">{html.escape(key)}</span>'
            f'<span class="tu-v">{html.escape(str(value))}</span></div>'
            for key, value in rows
        )
    else:
        body = '<p class="tu-hint">Your details appear here once we\'ve introduced ourselves.</p>'

    return f'<div id="tu-card"><h4>Student</h4>{body}</div>'


# --- Chat handler ------------------------------------------------------------
def _respond_core(message, history, profile):
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

    # A reasoning model can exhaust the token budget before emitting any visible
    # text. Retry once with far more room before giving up on the student.
    if not answer and not failed:
        try:
            for piece in stream_reply(messages, max_tokens=MAX_TOKENS * 3):
                answer += piece
                yield history[:-1] + [{"role": "assistant", "content": answer}], profile, ""
        except Exception as exc:  # noqa: BLE001
            failed = True
            answer = f"Sorry - I hit an error reaching the model.\n\n`{str(exc)[:300]}`"

    history[-1] = {
        "role": "assistant",
        "content": answer
        or (
            "I didn't manage to produce a reply there - the model returned no "
            "text. Please ask again, or split a long request into shorter ones."
        ),
    }
    if not failed and answer:
        log_telemetry(round(time.time() - started, 2), profile["major"])
    yield history, profile, ""


def respond(message, history, profile):
    """Wrap the core generator so the sidebar card tracks the profile."""
    for history_out, profile_out, box in _respond_core(message, history, profile):
        yield history_out, profile_out, box, render_student_card(profile_out)


def start_over():
    return (
        [{"role": "assistant", "content": WELCOME}],
        dict(DEFAULT_PROFILE),
        "",
        render_student_card(DEFAULT_PROFILE),
    )


def _launch_auth():
    user = os.environ.get("ADVISOR_USERNAME", "").strip()
    password = os.environ.get("ADVISOR_PASSWORD", "").strip()
    if user and password:
        return (user, password)
    print("[warn] ADVISOR_USERNAME/ADVISOR_PASSWORD not set - launching with no login.")
    return None


# --- UI ----------------------------------------------------------------------
with gr.Blocks(title="TUAdvisor") as ui:
    gr.HTML(
        '<div id="tu-header">'
        '<div class="tu-mark">TU</div>'
        "<div>"
        '<h1 class="tu-name">TUAdvisor</h1>'
        '<p class="tu-tag">Towson University &middot; Computer and Information Sciences</p>'
        "</div>"
        "</div>"
    )

    with gr.Row():
        with gr.Column(scale=3, min_width=430):
            chatbot = gr.Chatbot(
                value=[{"role": "assistant", "content": WELCOME}],
                height=530,
                show_label=False,
                line_breaks=True,
                elem_id="tu-chat",
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

        with gr.Column(scale=1, min_width=250):
            student_card = gr.HTML(render_student_card(DEFAULT_PROFILE))
            gr.HTML(
                '<div class="tu-note">Answers come from the official Towson catalog '
                "for the program you pick. Confirm anything binding with your "
                "department advisor.</div>"
            )

    profile = gr.State(dict(DEFAULT_PROFILE))

    outputs = [chatbot, profile, message, student_card]
    send.click(respond, [message, chatbot, profile], outputs)
    message.submit(respond, [message, chatbot, profile], outputs)
    reset.click(start_over, None, outputs)


if __name__ == "__main__":
    ui.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        auth=_launch_auth(),
        theme=THEME,
        css=CSS,
    )
