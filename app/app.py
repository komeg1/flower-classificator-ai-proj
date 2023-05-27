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
fileName = ''
def load_image():
    filename = tk.filedialog.askopenfilename(initialdir="/",
                                          title="Select a File")
    #load image from filename
    img = Image.open(filename)
    img = img.resize((300, 300))
    prepareImageToCheck(filename)
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

    #add prediction text to the window
    predictedText = ctk.CTkLabel(window, text=CLASSES[np.argmax(prediction)], font=('Lato', 15))
    predictedText.place(x=130, y=350)

    #add text that lists the probabilities of each class
    probabilitiesText = ctk.CTkLabel(window, text='Probabilities:', font=('Lato', 15))
    probabilitiesText.place(x=400, y=150)

    prediction = CLASSES[0] + ': ' + str(round(prediction[0][0]*100, 2)) + '%\n' + CLASSES[1] + ': ' + str(round(prediction[0][1] *100, 2)) + '%\n' + CLASSES[2] + ': ' + str(round(prediction[0][2]*100, 2)) + '%\n' + CLASSES[3] + ': ' + str(round(prediction[0][3]*100, 2)) + '%\n' + CLASSES[4] + ': ' + str(round(prediction[0][4]*100, 2)) + '%'


    probabilities = ctk.CTkLabel(window, text=prediction, font=('Lato', 15))
    probabilities.place(x=400, y=170)




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
    pass


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