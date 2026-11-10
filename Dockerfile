FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

WORKDIR /app

# Dependencies first so source edits do not invalidate the pip layer.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# The application source. Without these COPY lines the image contains no code
# and `python app/main.py` fails immediately.
COPY app/ ./app/
COPY data/ ./data/

# Telemetry appends to data/telemetry.csv, so the runtime user needs write access.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

CMD ["python", "app/main.py"]
