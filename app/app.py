import os
import tkinter as tk
from tkinter import ttk

import customtkinter
import cv2

from app_parameters import *
from PIL import Image, ImageTk
import customtkinter as ctk
import numpy as np
customtkinter.set_appearance_mode('light')
imageToCheck = []
def load_image():
    filename = tk.filedialog.askopenfilename(initialdir="/",
                                          title="Select a File")
    #load image from filename
    img = Image.open(filename)
    img = img.resize((300, 300))
    prepareImageToCheck(filename)
    print(imageToCheck)
    img = ImageTk.PhotoImage(img)
    # delete previous image
    for widget in imageFrame.winfo_children():
        widget.destroy()

    panel = tk.Label(imageFrame, image=img)

    panel.image = img
    panel.pack()

    return prepareImageToCheck(filename)

def prepareImageToCheck(filename):
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (150, 150))

    prediction = modelFile.predict(np.array([img])/255)
    print(prediction)

def chosenModel(event):

    if modelChoice.get() == 'Reference model':
        CHOSEN_MODEL = Model.REFERENCE
        print('Reference model')
    elif modelChoice.get() == 'Simple NN model':
        CHOSEN_MODEL = Model.SIMPLE
        print('Simple NN model')
    elif modelChoice.get() == 'CNN model':
        CHOSEN_MODEL = Model.CNN
        modelFile = load_model('cnn_model.h5')
        print('CNN model')

def runModel():
    print(imageToCheck)

    print(prediction)



window = ctk.CTk()
window.title("Flower classifier")
window.geometry('650x400')
window.resizable(False, False)




modelChoice = ctk.CTkComboBox(window, values=['Reference model', 'Simple NN model', 'CNN model'], command=chosenModel)
modelChoice.place(x=500, y=100)



modelChooseLabel = ctk.CTkLabel(window, text='Choose model:')
modelChooseLabel.place(x=410, y=100)

imageFrame = ctk.CTkFrame(window, width = 300, height = 300)
imageFrame.place(x=20,y =40)

#add button to upload a photo
uploadButton = ctk.CTkButton(window, text = 'Upload a photo', width = 20, height = 2, command=load_image)
uploadButton.place(x=250, y = 369)

runPredictionButton = ctk.CTkButton(window, text = 'Run prediction', width = 20, height = 2, command=runModel)
runPredictionButton.place(x=350, y = 369)

predictedTextLabel = ctk.CTkLabel(window, text='Predicted flower:', font=('Lato', 15))
predictedTextLabel.place(x=20, y=350)



window.mainloop()