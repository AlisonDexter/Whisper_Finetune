import pyaudio
import numpy as np
from queue import Queue
from threading import Thread
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
from huggingface_hub import HfApi, HfFolder, login
import os

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the fine-tuned Whisper model
MODEL_PATH = "D:/github/Whisper-Finetune-master/models/whisper-tiny-finetune"
processor = WhisperProcessor.from_pretrained(MODEL_PATH)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)
model.eval()

def login_hugging_face(token: str) -> None:
    """
    Log in to Hugging Face portal with a given token.
    """
    login(token)
    folder = HfFolder()
    folder.save_token(token)

    print('We are logged in to Hugging Face now')

os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"
token = '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'
login_hugging_face(token)
# Set your Hugging Face token as an environment variable
os.environ["HF_DATASETS_SERVER_TOKEN"] = "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"


# Create a pipeline for automatic speech recognition
asr_pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=0 if torch.cuda.is_available() else -1,
    chunk_length_s=30,
    
)
# 加载中文情感分析模型
#sentiment_pipeline = pipeline("sentiment-analysis", model="THUOCL/bert-base-chinese-sentiment")
pipe = pipeline("text-classification", model="hw2942/bert-base-chinese-finetuning-financial-news-sentiment-v2")

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms
# Initialize PyAudio
audio = pyaudio.PyAudio()
# Open audio stream
stream = audio.open(
    format=pyaudio.paFloat32,
    channels=1,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK
)
print("Listening…")
# Create a queue to store audio chunks
audio_queue = Queue()
# Function to continuously read audio from the microphone
def record_audio():
    while True:
        data = np.frombuffer(stream.read(CHUNK), dtype=np.float32)
        audio_queue.put(data)
# Start recording audio in a separate thread
record_thread = Thread(target=record_audio, daemon=True)
record_thread.start()
try:
    while True:
        # Collect 5 seconds of audio (50 chunks)
        audio_data = []
        for _ in range(50):
            audio_data.append(audio_queue.get())
        audio_data = np.concatenate(audio_data)
        # Transcribe the audio
        result = asr_pipe(audio_data)
        transcription = result['text'].strip()

        if transcription:
            print("Transcription:", transcription)

            # Perform sentiment analysis
            sentiment = pipe(transcription)[0]
            print("Sentiment:", sentiment['label'], f"(Score: {sentiment['score']:.2f})")
            print("—" * 30)

except KeyboardInterrupt:
    print("\nStopping…")

# Clean up
stream.stop_stream()
stream.close()
audio.terminate()
