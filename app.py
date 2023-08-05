import tkinter as tk
import customtkinter as ctk

import torch
import numpy as np
import pandas 

import cv2
from PIL import Image, ImageTk
import vlc
import random

app = tk.Tk()
app.geometry("600x600")
app.title("Drowsiness Detection")
ctk.set_appearance_mode("dark")


vidFrame = tk.Frame(height=480, width=600)
vidFrame.pack()
vid = ctk.CTkLabel(vidFrame)
vid.pack()

counter = 0
counterLabel = ctk.CTkLabel(master=app,text=counter,height=40, width=120, text_color="white", fg_color="teal")
counterLabel.pack(pady=10)

def resetcounter():
    global counter
    counter = 0
resetButton = ctk.CTkButton(master=app,text="Reset Counter", command = resetcounter,height=40, width=120, text_color="white", fg_color="teal")
resetButton.pack()


model = torch.hub.load('ultralytics/yolov5','custom', path='yolov5/runs/train/exp3/weights/last.pt', force_reload=True)
cap = cv2.VideoCapture(1)
def detect():
    global counter
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame)
    img = np.squeeze(results.render())
    
    if len(results.xywh[0]) > 0:
        dconf = results.xywh[0][0][4]
        dclass = results.xywh[0][0][5]
        
        if dconf.item() > 0.60 and dclass.item() == 16:
            p = vlc.MediaPlayer()
            q = vlc.Media("1.wav")
            p.set_media(q)
            p.play()
            counter += 1
    
    imgarr = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(imgarr)
    vid.imgtk = imgtk
    vid.configure(image=imgtk)
    vid.after(10,detect)
    counterLabel.configure(text=counter)
detect()
app.mainloop()