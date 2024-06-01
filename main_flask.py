from flask import Flask, request, jsonify  #type:ignore
import os
import time
import threading
from io import BytesIO

import google.generativeai as genai  #type:ignore
import pyaudio  #type:ignore
import speech_recognition as sr  #type:ignore
from PIL import Image  #type:ignore
import requests  #type:ignore
from PyPDF2 import PdfReader  #type:ignore
from dotenv import load_dotenv  #type:ignore

load_dotenv()

app = Flask(__name__)

api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

chat_session = model.start_chat(history=[])

loading = True

def display_loading():
    start_time = time.time()
    while loading:
        elapsed_time = time.time() - start_time
        seconds = int(elapsed_time)
        milliseconds = int((elapsed_time - seconds) * 1000)
        print(f"{seconds}.{milliseconds:03d}", end='\r')
        time.sleep(0.1)

def get_response(input_text):
    global loading
    loading_thread = threading.Thread(target=display_loading)
    loading_thread.start()
    response = chat_session.send_message(input_text)
    loading = False  
    loading_thread.join()  
    print("\nResponse received.")
    return response.text

def recognize_speech_from_mic():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Listening")
        audio = recognizer.listen(source)
    try:
        print("Recognizing")
        text = recognizer.recognize_google(audio)
        print(f"Recognized text: {text}")
        return text
    except sr.UnknownValueError:
        return "I did not understand that."
    except sr.RequestError:
        return "There was an error"

def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def generate_image(prompt):
    url = "https://api.gemini.ai/image/generate"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "prompt": prompt,
        "image_size": "512x512"
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        image_data = response.content
        with open("generated_image.png", "wb") as f:
            f.write(image_data)
        image = Image.open(BytesIO(image_data))
        image.show()
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")

def summarize_audio(file_path):
    your_file = genai.upload_file(path=file_path)

    prompt = "Listen carefully to the following audio file. Provide a brief summary."
    response = model.generate_content([prompt, your_file])

    return response.text

def analyze_video(file_path):
    print(f"Uploading file: {file_path}")
    video_file = genai.upload_file(path=file_path)
    print(f"Completed upload: {video_file.uri}")

    while video_file.state.name == "PROCESSING":
        print('Waiting..')
        time.sleep(10)
        video_file = genai.get_file(video_file.name)

    if video_file.state.name == "FAILED":
        raise ValueError(video_file.state.name)
    
    print(f'{video_file.uri}')

    prompt = "Describe this video."

    model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")

    print("")
    response = model.generate_content([prompt, video_file], request_options={"timeout": 600})
    return response.text

@app.route('/text', methods=['POST'])
def handle_text():
    data = request.json
    input_text = data.get('text')
    response_text = get_response(input_text)
    return jsonify({'response': response_text})

@app.route('/voice', methods=['POST'])
def handle_voice():
    input_text = recognize_speech_from_mic()
    response_text = get_response(input_text)
    return jsonify({'response': response_text})

@app.route('/pdf', methods=['POST'])
def handle_pdf():
    pdf_file = request.files['file']
    file_path = os.path.join("/tmp", pdf_file.filename)
    pdf_file.save(file_path)
    input_text = read_pdf(file_path)
    response_text = get_response(input_text)
    return jsonify({'response': response_text})

@app.route('/image', methods=['POST'])
def handle_image():
    data = request.json
    prompt = data.get('prompt')
    generate_image(prompt)
    return jsonify({'response': 'Image'})

@app.route('/audio', methods=['POST'])
def handle_audio():
    audio_file = request.files['file']
    file_path = os.path.join("/tmp", audio_file.filename)
    audio_file.save(file_path)
    response_text = summarize_audio(file_path)
    return jsonify({'response': response_text})

@app.route('/video', methods=['POST'])
def handle_video():
    video_file = request.files['file']
    file_path = os.path.join("/tmp", video_file.filename)
    video_file.save(file_path)
    response_text = analyze_video(file_path)
    return jsonify({'response': response_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
