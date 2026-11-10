FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir markitdown gradio instructor pydantic ollama openai
EXPOSE 7860
CMD ["python", "main.py"]
