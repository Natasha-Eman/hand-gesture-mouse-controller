import pyaudio
import numpy as np
import wave
import psutil
import winsound
from openwakeword.model import Model
from faster_whisper import WhisperModel

owwModel = Model(wakeword_models=["hey_jarvis"])
whisper_model = WhisperModel("small.en", device="cpu", compute_type="int8")

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1280
RECORD_SECONDS = 4

audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                     input=True, frames_per_buffer=CHUNK)

print("Listening for 'Hey Jarvis'... (Ctrl+C to stop)")

import subprocess
import pyautogui

def handle_command(text):
    text = text.lower()

    if "battery" in text:
        battery = psutil.sensors_battery()
        if battery:
            print(f"[ACTION] Battery is at {int(battery.percent)}%")
        else:
            print("[ACTION] Battery info not available (desktop PC?)")

    elif "volume up" in text:
        for _ in range(5):
            pyautogui.press("volumeup")
        print("[ACTION] Volume increased")

    elif "volume down" in text:
        for _ in range(5):
            pyautogui.press("volumedown")
        print("[ACTION] Volume decreased")

    elif "mute" in text:
        pyautogui.press("volumemute")
        print("[ACTION] Muted")

    elif "chrome" in text:
        subprocess.Popen(["start", "chrome"], shell=True)
        print("[ACTION] Opening Chrome")

    elif "notepad" in text:
        subprocess.Popen(["notepad.exe"])
        print("[ACTION] Opening Notepad")

    elif "scroll up" in text:
        pyautogui.scroll(300)
        print("[ACTION] Scrolled up")

    elif "scroll down" in text:
        pyautogui.scroll(-300)
        print("[ACTION] Scrolled down")

    else:
        print(f"[ACTION] No matching command for: \"{text}\"")

def record_command():
    winsound.Beep(1000, 150)
    print("Wake word detected! Listening for command...")
    frames = []
    num_chunks = int(RATE / CHUNK * RECORD_SECONDS)
    for _ in range(num_chunks):
        data = stream.read(CHUNK)
        frames.append(data)

    wf = wave.open("command.wav", "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))
    wf.close()

    segments, _ = whisper_model.transcribe("command.wav")
    text = " ".join([seg.text for seg in segments]).strip()
    print(f"You said: \"{text}\"")
    handle_command(text)
    winsound.Beep(1500, 100)

import time

COOLDOWN_SECONDS = 2
last_trigger_time = 0
armed = True   # only trigger when re-armed (score dropped back down)

try:
    while True:
        audio_chunk = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
        prediction = owwModel.predict(audio_chunk)

        now = time.time()

        for wakeword, score in prediction.items():
            if score > 0.5 and armed and (now - last_trigger_time) > COOLDOWN_SECONDS:
                armed = False
                record_command()
                last_trigger_time = time.time()   # cooldown starts AFTER recording finishes
            elif score < 0.2:
                armed = True   # re-arm only once score clearly drops

except KeyboardInterrupt:
    print("Stopped.")
    stream.stop_stream()
    stream.close()
    audio.terminate()

except KeyboardInterrupt:
    print("Stopped.")
    stream.stop_stream()
    stream.close()
    audio.terminate()

