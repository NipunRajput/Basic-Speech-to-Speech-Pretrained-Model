# Speech-to-Speech Chatbot with Face Detection

This project combines speech recognition, natural language processing, and face detection into a single application. It provides a Tkinter-based GUI for real-time speech-to-speech interaction with a chatbot while simultaneously running a face detection module using OpenCV.

---

## Features

- **Speech-to-Text**: Converts spoken input into text using the Google Speech Recognition API.
- **Chatbot**: Generates responses to user input using a pre-trained GPT-Neo language model.
- **Text-to-Speech**: Converts chatbot responses back into speech for a seamless interaction.
- **Face Detection**: Uses Haar cascades to detect faces in real-time via the webcam.
- **Graphical User Interface**: Built with Tkinter for user-friendly operation.

---

## Prerequisites

Make sure you have the following installed:

1. Python 3.8 or later
2. Pip package manager
3. GPU with CUDA support (optional, for better performance with the language model)

---

## Installation Steps

1. Clone the repository or download the source code.
   ```bash
   git clone https://github.com/your-repo/speech-chatbot-with-face-detection.git
   cd speech-chatbot-with-face-detection
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure that OpenCV's Haar cascades are available. These are typically included with OpenCV installations. The relevant file is `haarcascade_frontalface_default.xml`.

5. Download the pre-trained GPT-Neo model and tokenizer:
   ```bash
   from transformers import AutoTokenizer, AutoModelForCausalLM

   tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
   model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
   ```

---

## Running the Application

1. Start the application:
   ```bash
   python main.py
   ```

2. The GUI will appear, displaying a chat window and a "Speak" button.

3. Click the "Speak" button to initiate a conversation. Speak into the microphone to interact with the chatbot.

4. The webcam feed will display real-time face detection. Press `q` to exit the face detection window.

---

## How It Works

1. **Speech-to-Text**: Captures your spoken input via the microphone and converts it into text using the SpeechRecognition library.

2. **Natural Language Processing**: Processes the input text and generates a response using the GPT-Neo model.

3. **Text-to-Speech**: Converts the chatbot's response back into speech using the pyttsx3 library.

4. **Face Detection**: Continuously processes the webcam feed to detect faces using OpenCV's Haar cascades.

---

## Dependencies

- [SpeechRecognition](https://pypi.org/project/SpeechRecognition/): For speech-to-text conversion.
- [pyttsx3](https://pypi.org/project/pyttsx3/): For text-to-speech conversion.
- [Transformers](https://huggingface.co/transformers/): For GPT-Neo-based chatbot.
- [OpenCV](https://pypi.org/project/opencv-python/): For face detection.
- [Tkinter](https://docs.python.org/3/library/tkinter.html): For GUI development.
- [Torch](https://pytorch.org/): For running the GPT-Neo model.

---

## Notes

- The chatbot uses a smaller model (GPT-Neo 125M) for faster response time. For more complex applications, you can switch to a larger model.
- Ensure the microphone and webcam are functioning correctly before running the application.
- If you encounter any issues with speech recognition, check your internet connection as the Google Speech API requires it.

---

## Known Issues

- The application may lag on systems without GPU support.
- The face detection window must be closed manually by pressing `q`.

---

## License

This project is open-source and available under the MIT License. Feel free to use, modify, and distribute it.

---

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for providing pre-trained models.
- [OpenCV](https://opencv.org/) for face detection tools.
- [Python SpeechRecognition](https://pypi.org/project/SpeechRecognition/) for simplifying speech-to-text conversion.
