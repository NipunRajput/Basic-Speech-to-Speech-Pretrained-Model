import speech_recognition as sr
import pyttsx3
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import tkinter as tk
from tkinter import scrolledtext
import cv2
import threading

# Initialize Text-to-Speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Adjust the speech rate

# Load the pre-trained language model
model_name = "EleutherAI/gpt-neo-125M"  # Smaller, faster model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Ensure the tokenizer has a padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Warm-up the model
dummy_input = tokenizer("Hello!", return_tensors="pt", padding=True).to(device)
model.generate(dummy_input['input_ids'])

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Function to convert speech to text
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        try:
            log_message("Listening... Speak now.")
            audio = recognizer.listen(source, timeout=3)  # Reduced timeout
            text = recognizer.recognize_google(audio)
            log_message(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            log_message("Sorry, I could not understand the audio.")
            return ""
        except sr.RequestError as e:
            log_message(f"Error with speech recognition service: {e}")
            return ""

# Function to preprocess user input
def preprocess_input(text):
    text = text.lower().strip()
    corrections = {
        "what's": "what is",
        "i'm": "i am",
        "can't": "cannot",
        "you're": "you are"
    }
    for key, value in corrections.items():
        text = text.replace(key, value)
    return text

# Function to generate a chatbot response
def generate_response(prompt):
    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
        attention_mask = inputs['attention_mask']

        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=attention_mask,
            max_length=100,  # Reduced response length
            num_return_sequences=1,
            do_sample=True,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):].strip()  # Remove the prompt from the response
    except Exception as e:
        return f"Error: {e}"

# Function to convert text to speech
def text_to_speech(response):
    log_message(f"Chatbot: {response}")
    tts_engine.say(response)
    tts_engine.runAndWait()

# Function to log messages to the interface
def log_message(message):
    chat_display.insert(tk.END, message + "\n")
    chat_display.see(tk.END)

# Function to handle the conversation
def start_conversation():
    user_input = speech_to_text()
    if not user_input:
        return

    processed_input = preprocess_input(user_input)
    response = generate_response(processed_input)
    text_to_speech(response)

# Function to handle face detection
def face_detection():
    cap = cv2.VideoCapture(0)  # Open the camera
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale for Haar cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Display the video feed
        cv2.imshow("Face Detection", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Tkinter GUI
root = tk.Tk()
root.title("Speech-to-Speech Chatbot with Face Detection")

# Chat display
chat_display = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=20, width=50)
chat_display.pack(padx=10, pady=10)

# Button to start the chatbot
start_button = tk.Button(root, text="Speak", command=start_conversation, font=("Helvetica", 14))
start_button.pack(pady=10)

# Run face detection in a separate thread
face_thread = threading.Thread(target=face_detection, daemon=True)
face_thread.start()

# Start the GUI loop
root.mainloop()
