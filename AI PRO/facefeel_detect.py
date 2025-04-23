import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk

model = load_model('model/facefeel_model.keras')  


emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


cap = cv2.VideoCapture(0)


window = tk.Tk()
window.title("Emotion Detection")

label = Label(window)
label.pack()

emotion_label = Label(window, text="Emotion: ", font=("Arial", 20))
emotion_label.pack()

def start_detection():
    while True:
       
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

      
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
           
            face = frame[y:y + h, x:x + w]

            
            face = cv2.resize(face, (48, 48))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = face.astype('float32') / 255.0  
            face = np.reshape(face, (1, 48, 48, 1))  

            emotion_prediction = model.predict(face)
            max_index = np.argmax(emotion_prediction[0])
            predicted_emotion = emotion_labels[max_index]

            emotion_label.config(text=f"Emotion: {predicted_emotion}")

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = ImageTk.PhotoImage(img)

        label.config(image=img)
        label.image = img

        window.update_idletasks()
        window.update()


def stop_detection():
    cap.release()
    window.quit()


start_button = Button(window, text="Start Detection", font=("Arial", 14), command=start_detection)
start_button.pack(pady=10)

stop_button = Button(window, text="Stop Detection", font=("Arial", 14), command=stop_detection)
stop_button.pack(pady=10)

window.mainloop()
