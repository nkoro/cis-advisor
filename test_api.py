"""Manual connectivity check for the SiliconFlow endpoint.

Usage:
    SILICONFLOW_API_KEY=sk-... python test_api.py
"""

import os
import sys

import httpx
from openai import OpenAI

API_KEY = os.environ.get("SILICONFLOW_API_KEY", "").strip()
BASE_URL = os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.com/v1")
INSECURE_TLS = os.environ.get("SILICONFLOW_INSECURE_TLS", "").strip().lower() in {
    "1",
    "true",
    "yes",
}

if not API_KEY:
    sys.exit("SILICONFLOW_API_KEY is not set. See .env.example.")

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    http_client=httpx.Client(verify=not INSECURE_TLS),
)

try:
    models = client.models.list()
    print("SUCCESS! Connected to SiliconFlow. Available models count:", len(models.data))
except Exception as exc:
    print("FAILED:", exc)
