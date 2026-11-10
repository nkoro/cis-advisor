from openai import OpenAI
import httpx

client = OpenAI(
    base_url="https://api.siliconflow.cn/v1",
    api_key="REDACTED",
    http_client=httpx.Client(verify=False)
)

try:
    models = client.models.list()
    print("SUCCESS! Connected to SiliconFlow. Available models count:", len(models.data))
except Exception as e:
    print("FAILED:", e)
