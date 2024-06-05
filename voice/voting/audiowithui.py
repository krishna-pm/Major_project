import wave
import numpy as np
import os
import subprocess
import tkinter as tk
import pyaudio
import opensmile
import pandas as pd
import joblib
from PIL import ImageTk, Image

def record_audio():
    # Define the parameters for recording audio
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 20
    WAVE_OUTPUT_FILENAME = "recorded_audio.wav"
    
    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    # Start recording
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
    
    print("Recording started...")
    if (btn['text']=='Record Audio'):
        btn['text']='Recording'
    
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("Recording stopped.")

    

    # Stop recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded audio
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

def extract_features():
    # Use OpenSMILE to extract features from the audio
    #os.system("SMILExtract -C config/IS13_ComParE.conf -I recorded_audio.wav -O features.csv")
    file="recorded_audio.wav"
    smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.emobase,
    feature_level=opensmile.FeatureLevel.Functionals,
    )
    smile.feature_names
    feat = []
    print("processing file ... ", file)
    feat_i = smile.process_file(file)
    feat.append(feat_i.to_numpy().flatten())
    np.savetxt("features.csv", feat, delimiter=",")

def predict_output():
    filename = 'finalized_model.sav'
    model = joblib.load(filename)
    print("model loaded")
    # Load the features into a NumPy array
    data  = pd.read_csv('features.csv',sep= ',', header = None)
    features = data.values[:, 0:987]
    

    # Use a machine learning model to predict the output
   
    
    prediction = model.predict(features)
    return prediction

def show_output(output):
    # Display the prediction on the screen
    tk.Label(root, text=output).pack()

if __name__ == '__main__':
    root = tk.Tk()
    root.geometry("1000x800")
    root.title("Alzhemer's Detection")
    frame = tk.Frame(root, width=600, height=400)
    frame.pack()
    frame.place(anchor='center', relx=0.5, rely=0.5)
    img = ImageTk.PhotoImage(Image.open("../cookieTheftImage.png"))
    label = tk.Label(frame, image = img)
    label.pack()


    btn=tk.Button(root, text="Record Audio", command=record_audio)
    btn.pack()
    tk.Button(root, text="Extract Features", command=extract_features).pack()
    tk.Button(root, text="Predict Output", command=lambda: show_output(predict_output())).pack()
    root.mainloop()
