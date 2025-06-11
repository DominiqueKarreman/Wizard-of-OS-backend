from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import os
from flask_jwt_extended import JWTManager, get_jwt_identity, jwt_required
app = Flask(__name__)
from merlin import answerPromptStream, optimize_week_concurrently, Event, generate_week_summary, AskRequest, SimpleEvent
import uuid
from datetime import datetime   
from ollama import chat

# Enable cors on the server
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['JWT_SECRET_KEY'] = 'ikhouvankaasenikhouvanpsemo2025'
@app.route('/')
def home():
    return "Welcome to the Flask API!"

@app.route('/prompt/text', methods=['POST'])
def promptPostStream():
    file_path = None
    if request.content_type.startswith('multipart/form-data'):
        # Handle multipart/form-data (for prompt and file)
        print("we got a multipart request")
        args = request.form
        if 'file' in request.files and 'prompt' in args:
            prompt = args['prompt']
            if 'model' in args:
                model = args['model']
                print("model: ", model)
            else:
                model = "llama3"
            if 'clipboard' in args and 'clipboardContext' in args:
                clipboardContext = args['clipboardContext']
                if clipboardContext == True:
                    clipboard = args['clipboard']
                    print("clipboard: ", clipboard)
                else: 
                    clipboard = ""
            else:
                clipboard = ""
            # Handle the file if needed
            print('file in request')
            print(request.files)
            try:
                file = request.files['file']
                # Save the file to the 'uploads' directory
                file_path = os.path.join(r'./visionPromptImages', file.filename)
                print("file_path: ", file_path)
                file.save(file_path)
            except Exception as e:
                return f"Error: {e}"
        else:
            return jsonify({"error": "no prompt or file detected"}), 400 
    elif request.content_type == 'application/json':
        # Handle JSON (for prompt without a file)
        json_data = request.get_json()
        if 'prompt' in json_data:
            prompt = json_data['prompt']
        else:
            return jsonify({"error": "no prompt detected"}), 400
        if 'model' in json_data:
            model = json_data['model']
            print("model: ", model)
        else:
            model = "llama3"
        if 'clipboard' in json_data and 'clipboardContext' in json_data:
            clipboardContext = json_data['clipboardContext']
            if clipboardContext == True:
                clipboard = json_data['clipboard']
                print("clipboard: ", clipboard)
            else: 
                clipboard = ""
        else:
            clipboard = ""
    else:
        return jsonify({"error": "Unsupported content type"}), 400

    # Pass the prompt and (optional) image context to the assistant's streaming function
    return Response(answerPromptStream(prompt, model, file_path, clipboard), mimetype='text/event-stream')


# from optimize_utils import Event, optimize_week_concurrently

@app.route('/optimize', methods=['POST'])
def receive_events():
    try:
        raw_events = request.get_json()
        if not raw_events:
            return jsonify({"error": "No JSON received"}), 400

        events = [Event(**event) for event in raw_events]
        optimized = optimize_week_concurrently(events)
        response_data = {
            "status": "success",
            "optimized": optimized
        }

        print("hi", response_data)
        return jsonify({
            "status": "success",
            "optimized": optimized
        }), 200

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"error": str(e)}), 500

@app.route('/week-summary', methods=['POST'])
def week_summary():
    try:
        raw_events = request.get_json()
        print(raw_events)
        if not raw_events:
            return jsonify({"error": "No JSON received"}), 400

        events = [Event(**event) for event in raw_events]
        print("how many events:", len(events))
        text_summary = generate_week_summary(events)  # This should return a string
        print(text_summary)
        return jsonify({
            "status": "success",
            "summary": text_summary
        }), 200

    except Exception as e:
        print("❌ Error in /week-summary:", e)
        return jsonify({"error": str(e)}), 500


import json

system_prompt_qa = (
    "You are a helpful assistant specialized in analyzing weekly schedules. "
    "Based on the user's calendar events, answer their question with relevant details. "
    "Consider realistic human behavior—e.g., people don’t run during meetings or lunch at 6am. "
    "Be concise, insightful, and kind. "
    "Always base your answer on the provided schedule, and do not assume events that aren't listed."
)
# Flask route
@app.route('/planning/ask', methods=['POST'])
def ask_schedule_question():
    try:
        body = request.get_json()
        parsed = AskRequest(**body)

        events_json = json.dumps([e.model_dump() for e in parsed.events])
        question = parsed.question

        messages = [
            {"role": "system", "content": system_prompt_qa},
            {"role": "user", "content": f"Here is my schedule: {events_json}\n\nMy question: {question}"}
        ]

        response = chat(model="llama3.1", messages=messages)
        answer = response["message"]["content"]

        return jsonify({"answer": answer}), 200

    except Exception as e:
        return jsonify({"error": f"Failed to process question: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(port=5002, debug=True)

    # Python (Flask) - AI Planning Commentary Endpoint
