from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict
import json
from pydantic import BaseModel, RootModel
import os
import mimetypes
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd
from PIL import Image
import requests


class OllamaClient:
    def __init__(self, base_url="http://host.docker.internal:11434"):
        self.base_url = base_url.rstrip('/')

    def chat(self, model, messages, stream=False, format=None):
        url = f"{self.base_url}/v1/chat/completions"
        payload = {"model": model, "messages": messages}
        if format:
            payload["format"] = format
        if stream:
            with requests.post(url, json=payload, stream=True) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    dec = line.decode('utf-8')
                    if dec.startswith("data: "):
                        data = dec[len("data: "):]
                        if data.strip() == "[DONE]":
                            break
                        yield json.loads(data)
        else:
            resp = requests.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()

ollama_client = OllamaClient(base_url="http://192.168.2.27:11434")

# System prompts
system_prompt_default = (
    "You are MERLIN, an AI designed to assist users in their tasks. "
    "Treat every new prompt as a new conversation..."
)

system_prompt_clipboard = (
    "You are MERLIN, an AI designed to assist users in their tasks. "
    "In each interaction, you will use the clipboard text..."
)

system_prompt_planner = (
    "You are THE ULTIMATE PLANNING AI..."
)

system_prompt_qa = (
    "You are a helpful assistant specialized in analyzing weekly schedules..."
)

convo = [{'role': 'system', 'content': system_prompt_default}]

class Event(BaseModel):
    title: str
    startDate: str
    endDate: str
    location: str | None = None
    notes: str | None = None
    calendar: str | None = None
    url: str | None = None
    organizerName: str | None = None
    organizerEmail: str | None = None
    isAllDay: bool = False

class OptimizedEvents(RootModel[List[Event]]): pass

class SimpleEvent(BaseModel):
    title: str
    startDate: str
    endDate: str

class AskRequest(BaseModel):
    question: str
    events: List[SimpleEvent]

def extract_text_from_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.pdf':
            reader = PdfReader(file_path)
            return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
        elif ext == '.docx':
            doc = Document(file_path)
            return "\n".join(paragraph.text for paragraph in doc.paragraphs)
        elif ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
            return df.to_string(index=False)
        elif ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif']:
            return None
        else:
            return "Unsupported file type."
    except Exception as e:
        return f"Error reading file: {e}"

def is_image_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    return ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif']

def vision_prompt(prompt, file_path):
    res = ollama_client.chat(
        model="llava",
        messages=[{
            'role': 'user',
            'content': f'You are the vision analysis AI...\nUSER PROMPT: {prompt}',
            'images': [file_path]
        }]
    )
    return res['message']['content']

def answerPromptStream(prompt, model, file_path, clipboard):
    system_prompt = system_prompt_clipboard if clipboard else system_prompt_default
    image_context = None
    file_context = None

    if file_path:
        if is_image_file(file_path):
            try:
                image_context = vision_prompt(prompt, file_path)
            except Exception as e:
                image_context = "Image processing failed."
        else:
            file_context = extract_text_from_file(file_path)

    if not convo or convo[0]['role'] == 'system':
        convo.insert(0, {'role': 'system', 'content': system_prompt})

    prompt_parts = [f"USER PROMPT: {prompt}"]
    if image_context:
        prompt_parts.append(f"IMAGE CONTEXT: {image_context}")
    if file_context:
        prompt_parts.append(f"FILE CONTEXT: {file_context}")
    if clipboard:
        prompt_parts.append(f"CLIPBOARD CONTEXT: {clipboard}")

    full_prompt = ". ".join(prompt_parts)
    convo.append({'role': 'user', 'content': full_prompt})

    response = ''
    try:
        stream = ollama_client.chat(model=model, messages=convo, stream=True)
        for chunk in stream:
            delta = chunk["choices"][0]["delta"]
            content = delta.get("content", "")
            response += content
            content = content.replace('\n', '__NEWLINE__')
            yield f"data: {content}\n\n"
    except Exception as e:
        yield f"data: ERROR: {e}\n\n"

def group_events_by_day(events: List[Event]) -> Dict[str, List[Event]]:
    days: Dict[str, List[Event]] = {}
    for event in events:
        day = event.startDate[:10]
        days.setdefault(day, []).append(event)
    return days

def process_day_events(day: str, day_events: List[Event]) -> List[Dict]:
    json_data = json.dumps([e.model_dump() for e in day_events])
    messages = [
        {'role': 'system', 'content': system_prompt_planner},
        {'role': 'user', 'content': json_data}
    ]
    try:
        response = ollama_client.chat(model="llama3.1", messages=messages, format="json")
        content = response['choices'][0]['message']['content']
        data = json.loads(content)

        if isinstance(data, dict):
            if 'optimizedEvents' in data and isinstance(data['optimizedEvents'], list):
                data = data['optimizedEvents']
            else:
                data = [data]

        optimized = OptimizedEvents.model_validate(data)
        return [e.model_dump() for e in optimized.root]
    except Exception as e:
        print(f"❌ Failed to optimize events for {day}:", e)
        return []

def optimize_week_concurrently(events: List[Event]) -> List[Dict]:
    grouped = group_events_by_day(events)
    results: List[Dict] = []
    with ThreadPoolExecutor(max_workers=7) as executor:
        futures = {
            executor.submit(process_day_events, day, day_events): day
            for day, day_events in grouped.items()
        }
        for future in futures:
            results.extend(future.result())
    return results

def generate_week_summary(events: List[Event]) -> str:
    prompt = (
        "You are an AI planning assistant. Given the following list of events in JSON format, "
        "summarize the user's weekly schedule in a helpful, motivating tone."
    )
    messages = [
        {'role': 'system', 'content': prompt},
        {'role': 'user', 'content': json.dumps([e.model_dump() for e in events])}
    ]
    response = ollama_client.chat(model="llama3.1", messages=messages)
    return response['choices'][0]['message']['content']