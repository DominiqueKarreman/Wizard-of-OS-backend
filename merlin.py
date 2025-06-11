from ollama import chat  # Ensure the ollama module is installed and accessible
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict
import json
from pydantic import BaseModel, RootModel

# Base system prompts
system_prompt_default = (
    "You are MERLIN, an AI designed to assist users in their tasks. "
    "Treat every new prompt as a new conversation. Do not refer to previous prompts unless they are relevant to the current conversation. "
    "You are a conversational AI that can provide information, answer questions, and perform tasks. "
    "You are designed to be helpful, friendly, and informative. "
    "You are not a human, but you are designed to communicate in a way that is natural and easy to understand. "
    "Only talk about previous prompts if they are relevant to the current conversation. "
    "If the user asks a question, provide a clear and concise answer."
)

system_prompt_clipboard = (
    "You are MERLIN, an AI designed to assist users in their tasks. "
    "In each interaction, you will use the clipboard text passed in by the user as context for your responses. "
    "The clipboard text should serve as a reference point to better understand the user's needs, "
    "and you should incorporate relevant details from it into your responses to provide more accurate and helpful assistance. "
    "Your goal is to make the user feel heard and understood by drawing on the clipboard content to help answer questions, "
    "clarify information, or guide the user in completing tasks. "
    "While doing so, ensure that your responses are clear, concise, and friendly. "
    "You are not a human, but your interactions should be natural, engaging, and easy to understand. "
    "You should only refer to the clipboard context if it helps with the current conversation. "
    "If the user provides new information or context, update your responses accordingly, without referring to irrelevant prior context unless it directly applies."
)
convo = [{'role': 'system', 'content': system_prompt_default}]

import os
import mimetypes

from PyPDF2 import PdfReader
from docx import Document
import pandas as pd
from PIL import Image

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
            return None  # Will be handled by vision_prompt

        else:
            return "Unsupported file type."
    except Exception as e:
        return f"Error reading file: {e}"

# Function to handle conversation
def is_image_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    return ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif']

def answerPromptStream(prompt, model, file_path, clipboard):
    if clipboard != "":
        print("Using clipboard context", clipboard)
        system_prompt = system_prompt_clipboard
    else:
        print("No clipboard context provided")
        system_prompt = system_prompt_default

    # Initialize contexts
    image_context = None
    file_context = None

    # File handling
    if file_path:
        if is_image_file(file_path):
            try:
                image_context = vision_prompt(prompt, file_path)
                print(image_context, " image context \n\n")
            except Exception as e:
                print("Error in vision_prompt:", e)
                image_context = "Image processing failed."
        else:
            file_context = extract_text_from_file(file_path)
            print(file_context, " file context \n\n")

    # Insert system prompt if needed
    if not convo or convo[0]['role'] == 'system':
        convo.insert(0, {'role': 'system', 'content': system_prompt})

    # Build final prompt
    prompt_parts = [f"USER PROMPT: {prompt}"]
    if image_context:
        prompt_parts.append(f"IMAGE CONTEXT: {image_context}")
    if file_context:
        prompt_parts.append(f"FILE CONTEXT: {file_context}")
    if clipboard:
        prompt_parts.append(f"CLIPBOARD CONTEXT: {clipboard}")

    full_prompt = ". ".join(prompt_parts)

    print(f'USER: {full_prompt}')
    convo.append({'role': 'user', 'content': full_prompt})

    response = ''
    try:
        stream = chat(model=model, messages=convo, stream=True)
        print('\nASSISTANT:')

        for chunk in stream:
            content = chunk['message']['content']
            response += content
            print(content, end='', flush=True)
            content = content.replace('\n', '__NEWLINE__')
            yield f"data: {content}\n\n"

        print('\n')

    except Exception as e:
        print(f"Streaming error: {e}")
        yield f"data: ERROR: {e}\n\n"
    
    
def vision_prompt(prompt, file_path):
    """
    Generate a vision prompt response for the given prompt and file path.
    
    Args:
        prompt (str): The user's prompt.
        file_path (str): The path to the image file.
    
    Returns:
        response (str): The assistant's response based on the image context.
    """
    res = chat(
        model="llava",
        messages=[
            {
                'role': 'user',
                'content': 'You are the vision analysis AI that provides semantic meaning from images to provide context '
                'to send to another AI that will create a response to the user. Do not respond as the AI assistant '
                'to the user. Instead take the user prompt input and try to extract all meaning from the photo '
                'relevant to the user prompt. Then generate as much objective data about the image for the AI '
                f'assistant who will respond to the user. \nUSER PROMPT: {prompt}',
                'images': [f'{file_path}']
            }
        ]
    )
    return res['message']['content']





system_prompt_planner = (
    "You are THE ULTIMATE PLANNING AI — an expert time architect designed to optimize a human's schedule for maximum productivity, realism, and well-being. "
    "Your goal is to restructure daily events into a plan that is efficient, human-centered, and achievable. "
    "You receive JSON data containing calendar events for a given day. Based on this, you must respond with a single JSON array of updated or reordered events for that day — with no commentary, explanation, or metadata. "
    "\n\n"
    "Here are the key planning principles and rules you must follow:\n"
    "\n"
    "📅 GENERAL PLANNING RULES:\n"
    "- Always maintain the realism of the day. Do not schedule overlapping events.\n"
    "- Never schedule back-to-back long tasks (>90 min) without a 15-30 min break.\n"
    "- All tasks must start and end within the same day.\n"
    "- Preserve original event durations unless absolutely necessary.\n"
    "- Do not move fixed events that are clearly external (e.g., meetings or events with organizers).\n"
    "\n"
    "☀️ MORNING PRIORITIZATION:\n"
    "- Prioritize deep work, long tasks, and focus sessions between 8:00 and 11:30.\n"
    "- Avoid scheduling any 'light' tasks like email or admin first thing in the morning.\n"
    "- Do not schedule lunch or social events before 11:30.\n"
    "\n"
    "🌙 EVENING RULES:\n"
    "- Physical activities like 'going for a run' or 'workout' are ideal between 20:00 and 23:00.\n"
    "- Avoid mentally intense work past 20:00 unless explicitly indicated by the task title.\n"
    "- Do not schedule any new tasks after 23:00 unless they are labeled 'late night' or 'urgent'.\n"
    "\n"
    "🍽️ HUMAN BEHAVIOR:\n"
    "- Never schedule meals before 7:00 or after 21:00 unless specified (e.g., 'late dinner').\n"
    "- Try to place 'lunch' between 12:00 and 13:30, and 'dinner' between 18:00 and 20:00.\n"
    "- Respect human breaks: After 2 hours of consecutive tasks, schedule at least a 15-minute gap.\n"
    "\n"
    "🚫 CONFLICTS & REDUNDANCY:\n"
    "- Do not schedule two events at the same time.\n"
    "- Do not schedule two events with the same title on the same day.\n"
    "- Avoid rescheduling a task to overlap with an existing commitment, especially meetings or external events.\n"
    "\n"
    "📌 CONTEXT-AWARENESS:\n"
    "- Titles like 'meeting', 'call', or events with an organizerEmail should be treated as fixed and non-movable.\n"
    "- Events marked as isAllDay should remain unchanged.\n"
    "- Only modify flexible or user-created events without external organizers.\n"
    "\n"
    "🧠 COGNITIVE LOAD:\n"
    "- Alternate between high-focus and low-focus tasks where possible.\n"
    "- Do not schedule more than 3 long (90+ min) sessions in one day.\n"
    "- Try not to end the day with heavy cognitive work — prefer reflection, review, or light activities after 20:00.\n"
    "\n"
    "🎯 YOUR OUTPUT:\n"
    "- Always return a single, clean JSON array of optimized events (no explanation).\n"
    "- All times must be ISO 8601 format (e.g., 2025-07-21T13:00:00Z).\n"
    "- Maintain the original fields of each event, but update startDate and endDate as needed.\n"
    "\n"
    "Example output: [ {\"title\": \"Work session\", \"startDate\": \"2025-07-21T09:00:00Z\", \"endDate\": \"2025-07-21T10:30:00Z\", ... }, {...} ]\n"
)

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
        response = chat(
            model="llama3.1",
            messages=messages,
            format="json"
        )
        content = response['message']['content']
        data = json.loads(content)

        # Case 1: wrapped in a key
        if isinstance(data, dict):
            if 'optimizedEvents' in data and isinstance(data['optimizedEvents'], list):
                data = data['optimizedEvents']
            else:
                data = [data]  # maybe it's a single event object

        # Now data is a list of dicts
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

    response = chat(model="llama3.1", messages=messages)
    return response['message']['content']


# System prompt for the schedule Q&A
system_prompt_qa = (
    "You are a helpful assistant specialized in analyzing weekly schedules. "
    "Based on the user's calendar events, answer their question with relevant details. "
    "Consider realistic human behavior—e.g., people don’t run during meetings or lunch at 6am. "
    "Be concise, insightful, and kind. "
    "Always base your answer on the provided schedule, and do not assume events that aren't listed."
)

# Define input models
class SimpleEvent(BaseModel):
    title: str
    startDate: str  # ISO format
    endDate: str

class AskRequest(BaseModel):
    question: str
    events: List[SimpleEvent]

