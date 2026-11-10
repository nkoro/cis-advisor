import gradio as gr
import instructor
from openai import OpenAI
from pydantic import BaseModel
from typing import List
import time
import os
import csv
import httpx
import warnings

# Suppress the InsecureRequestWarning so your terminal stays clean during the demo
warnings.filterwarnings("ignore", message="Unverified HTTPS request")

# 1. Define the Structured Output Schema
class SemesterPlan(BaseModel):
    semester_name: str
    courses: List[str]
    total_credits: int

class DegreePlanDraft(BaseModel):
    remaining_requirements_met: bool
    plan: List[SemesterPlan]
    advisor_notes: str

# 2. Setup Instructor with SiliconFlow API and SSL Bypass
api_key = "REDACTED" # <-- PASTE YOUR REAL KEY HERE

client = instructor.from_openai(
    OpenAI(
        base_url="https://api.siliconflow.com",
        api_key=api_key, 
        http_client=httpx.Client(verify=False) # Bypasses the university SSL inspection
    ),
    mode=instructor.Mode.JSON,
)

# 3. The Core Advisory Function
def generate_plan(name, student_id, major, courses_taken, preferences):
    start_time = time.time()
    
    # Hard Routing: Load exactly the markdown file needed
    file_path = f"data/{major.replace(' ', '_').lower()}.md"
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            major_context = f.read()
    except FileNotFoundError:
        return f"Error: Context file for {major} not found."

    system_prompt = f"""You are a strict academic advisor for Towson University.
    Your sole purpose is to draft Degree Completion Plans based EXACTLY on the provided context.
    
    STRICT RULES:
    1. Do not answer questions outside of degree planning.
    2. If you detect an SSN or Credit Card in the input, output: "ERROR: Sensitive PII detected."
    3. Verify prerequisites are met before placing a course in a future semester.
    
    CATALOG CONTEXT:
    {major_context}
    """
    
    user_prompt = f"Student: {name} ({student_id})\nTaken/Taking: {courses_taken}\nPreferences: {preferences}"

    try:
        resp = client.chat.completions.create(
            model="Qwen/Qwen2.5-72B-Instruct", 
            response_model=DegreePlanDraft,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1, # Keep it deterministic
        )
    except Exception as e:
        return f"Inference Error: {str(e)}"
    
    latency = round(time.time() - start_time, 2)
    
    # 4. Telemetry Logging for Research
    with open("data/telemetry.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([time.time(), latency, major])

    # 5. Format Output for the UI
    output = f"### Plan for {name} ({student_id})\n**Status:** {'On Track' if resp.remaining_requirements_met else 'Missing Requirements'}\n\n"
    for sem in resp.plan:
        output += f"**{sem.semester_name}** ({sem.total_credits} credits)\n"
        for course in sem.courses:
            output += f"- {course}\n"
        output += "\n"
    output += f"**Advisor Notes:**\n{resp.advisor_notes}\n\n*Generated in {latency} seconds using SiliconFlow (Qwen2.5-72B)*"
    
    return output

# 6. Build the Gradio UI
with gr.Blocks() as ui:
    gr.Markdown("# CIS Degree Plan Advisor")
    with gr.Row():
        with gr.Column():
            name = gr.Textbox(label="Student Name")
            student_id = gr.Textbox(label="Student ID")
            major = gr.Dropdown(
                choices=[
                    "Information Technology", 
                    "Computer and Mathematical Sciences", 
                    "Information Systems",
                    "Accelerated BS MS Computer Science",
                    "Computer Science",
                    "Computer and Mathematical Sciences Secondary Education"
                ], 
                label="Select Major"
            )
            courses = gr.Textbox(label="Courses Taken/Taking (Comma separated)", lines=3)
            prefs = gr.Textbox(label="Schedule Preferences (e.g., Max 15 credits, no mornings)", lines=2)
            submit = gr.Button("Generate Plan")
            
        with gr.Column():
            output_display = gr.Markdown(label="Generated Plan")
            
    submit.click(fn=generate_plan, inputs=[name, student_id, major, courses, prefs], outputs=output_display)

if __name__ == "__main__":
    ui.launch(server_name="0.0.0.0", server_port=7860, auth=("professor", "towson2026"))
