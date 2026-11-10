# TUAdvisor

A conversational academic advisor for Towson University's Department of Computer
and Information Sciences. Students check in through chat (name, student ID,
program), then talk to an advisor that answers only from the official catalog.

## How it works

```
Student --> Gradio chat UI --> respond() --> SiliconFlow chat completions (streaming)
                                   |
                                   +--> intake state machine (name -> ID -> program)
                                   +--> data/<major>.md    (catalog grounding)
                                   +--> data/telemetry.csv (latency research log)
```

`respond()` runs the intake state machine first, then streams the advisor reply.
The chosen program maps to exactly one markdown catalog file, injected as the
only source of truth in the system prompt -- no vector store, no retrieval step.

## Model choice

`deepseek-ai/DeepSeek-V4.1-Flash`, measured on this app's real workload (a
~19k-token catalog prompt):

| Model | Latency | Prompt cached | Cost / turn |
| --- | --- | --- | --- |
| **DeepSeek-V4.1-Flash** | **1.2-1.5 s** | 19,200 / 19,427 tok | **~$0.00013** |
| GLM-5.3-Flash | 3.4-4.0 s | 20,224 / 20,305 tok | ~$0.00066 |
| gpt-oss-20b | 2.1-2.3 s | none | ~$0.00080 |

DeepSeek lists at $0.15 input / $0.003 cached / $0.60 output per 1M tokens. The
cache discount is why the model with the cheapest *headline* input price is not
the cheapest in practice here. Override with `SILICONFLOW_MODEL`.

## Quick start (local)

```bash
python -m venv cis_env
source cis_env/bin/activate
pip install -r requirements.txt

cp .env.example .env      # then set SILICONFLOW_API_KEY
set -a; source .env; set +a

python app/main.py        # http://localhost:7860
```

Check API connectivity without starting the UI:

```bash
python test_api.py
```

## Docker

```bash
cp .env.example .env      # then set SILICONFLOW_API_KEY
docker compose up --build
```

The image installs `requirements.txt`, copies `app/` and `data/`, and runs as a
non-root user. `./data` is bind-mounted so telemetry survives rebuilds. No local
GPU is required because inference happens at SiliconFlow.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `SILICONFLOW_API_KEY` | *(required)* | SiliconFlow API key. Without it the UI reports a configuration error instead of calling the API. |
| `SILICONFLOW_BASE_URL` | `https://api.siliconflow.com/v1` | API base URL. |
| `SILICONFLOW_MODEL` | `deepseek-ai/DeepSeek-V4.1-Flash` | Chat model. |
| `SILICONFLOW_TIMEOUT` | `180` | Per-request timeout, seconds. |
| `SILICONFLOW_MAX_TOKENS` | `900` | Max tokens per reply. |
| `ADVISOR_HISTORY_TURNS` | `12` | Past messages resent to the model. |
| `SILICONFLOW_INSECURE_TLS` | `0` | Set to `1` only if your network does TLS inspection with an untrusted CA. |
| `ADVISOR_USERNAME` / `ADVISOR_PASSWORD` | *(unset)* | Gradio basic-auth login. **If unset the UI has no login.** |
| `GRADIO_SERVER_NAME` / `GRADIO_SERVER_PORT` | `0.0.0.0` / `7860` | Gradio bind address. |

## Repository layout

```
cis-advisor/
├── app/
│   ├── main.py     # Gradio UI, prompt, SiliconFlow routing, telemetry
│   └── ingest.py   # rebuilds data/*.md from catalog.towson.edu (markitdown)
├── data/           # catalog context, one markdown file per major
├── tests/          # offline pytest suite (no API key needed)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── requirements-dev.txt
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

The suite stubs the model client, so it runs offline and never spends tokens.

## Refreshing the catalog

`app/ingest.py` re-downloads each program page from `catalog.towson.edu` and
rewrites `data/*.md`. Run it from the repository root:

```bash
python app/ingest.py
```

## Security notes

* The API key is read from the environment only. If a key was ever committed,
  rotate it in the SiliconFlow console and purge it from git history
  (`git filter-repo`); deleting it from `HEAD` is not enough, because the old
  commits still contain it.
* TLS verification is on by default. `SILICONFLOW_INSECURE_TLS=1` is an explicit
  opt-out for campus SSL inspection, not the default.
* SSNs and Luhn-valid card numbers are rejected in code before any API call, so
  PII never leaves the process.

## Deployment

The app is a standard Gradio process, so it runs anywhere Python or Docker runs.
The Docker path above is the supported deployment: `docker compose up --build`.
No GPU, no external model host, and no third-party app platform is required --
inference is a plain HTTPS call to the SiliconFlow API.
